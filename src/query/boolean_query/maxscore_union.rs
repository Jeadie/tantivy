use super::block_wand_union::{block_wand, block_wand_single_scorer};
use crate::query::term_query::TermScorer;
use crate::query::Scorer;
use crate::{DocId, DocSet, Score, TERMINATED};

/// Term scorers ordered by increasing term-wide score bound.
struct BoundedScorers {
    scorers: Vec<TermScorer>,
    /// `prefix_bounds[i]` is the sum of the term-wide bounds of `scorers[..i]`. It has
    /// `scorers.len() + 1` entries.
    prefix_bounds: Vec<Score>,
}

impl BoundedScorers {
    fn new(mut scorers: Vec<TermScorer>) -> Self {
        scorers.sort_by(|left, right| left.max_score().total_cmp(&right.max_score()));
        let mut bounded_scorers = BoundedScorers {
            scorers,
            prefix_bounds: Vec::new(),
        };
        bounded_scorers.update_bounds();
        bounded_scorers
    }

    fn update_bounds(&mut self) {
        self.prefix_bounds.clear();
        let mut sum = 0.0;
        self.prefix_bounds.push(sum);
        for scorer in &self.scorers {
            sum += scorer.max_score();
            self.prefix_bounds.push(sum);
        }
    }

    /// Removes the exhausted scorers. This is called only when the partition is recomputed.
    fn remove_exhausted(&mut self) {
        let num_scorers = self.scorers.len();
        self.scorers.retain(|scorer| scorer.doc() != TERMINATED);
        if self.scorers.len() != num_scorers {
            self.update_bounds();
        }
    }

    /// Number of non-essential scorers: the longest prefix of `scorers` whose summed bound
    /// cannot exceed `threshold`. A document that matches only these scorers cannot be
    /// competitive, so they do not need to generate candidates.
    fn num_non_essential(&self, threshold: Score) -> usize {
        self.prefix_bounds[1..].partition_point(|&prefix_bound| prefix_bound <= threshold)
    }
}

/// Implements the MaxScore algorithm for dynamic pruning of a union of term scorers whose
/// scores are summed.
///
/// Scorers are ordered by their term-wide score bound. The longest prefix whose summed
/// bound cannot exceed the threshold is "non-essential": only the remaining "essential"
/// scorers generate candidate documents. Non-essential scorers are only probed to complete
/// the score of a candidate, from the highest bound to the lowest, and probing stops when the
/// remaining bounds cannot lift the candidate above the threshold.
///
/// Documents are passed to `callback` in increasing order, with their full score, when that
/// score exceeds the threshold. `callback` returns the new threshold.
///
/// Reference: Turtle and Flood, "Query evaluation: strategies and optimizations" (1995).
pub fn maxscore(
    mut scorers: Vec<TermScorer>,
    mut threshold: Score,
    callback: &mut dyn FnMut(DocId, Score) -> Score,
) {
    scorers.retain(|scorer| scorer.doc() != TERMINATED);
    if scorers.len() <= 1 {
        if let Some(scorer) = scorers.pop() {
            block_wand_single_scorer(scorer, threshold, callback);
        }
        return;
    }
    // The essential/non-essential partition requires non-negative bounds (e.g. no negative
    // boost). Keep the existing behaviour for other cases.
    if !scorers.iter().all(|scorer| scorer.max_score() >= 0.0) {
        block_wand(scorers, threshold, callback);
        return;
    }

    let mut bounded_scorers = BoundedScorers::new(scorers);
    // All documents before `floor` were already evaluated.
    let mut floor: DocId = 0;
    // `docs[i]` caches the current doc of the i-th essential scorer, so that finding the next
    // candidate scans a contiguous slice.
    let mut docs: Vec<DocId> = Vec::new();
    // Outer loop: (re)compute the essential/non-essential partition. This only happens when
    // a scorer is exhausted or the threshold moves enough to change the partition.
    loop {
        bounded_scorers.remove_exhausted();
        match bounded_scorers.scorers.len() {
            0 => return,
            1 => {
                let mut scorer = bounded_scorers.scorers.pop().unwrap();
                if scorer.doc() < floor {
                    scorer.seek(floor);
                }
                block_wand_single_scorer(scorer, threshold, callback);
                return;
            }
            _ => {}
        }
        let num_non_essential = bounded_scorers.num_non_essential(threshold);
        let BoundedScorers {
            scorers,
            prefix_bounds,
        } = &mut bounded_scorers;
        // The partition gains one more non-essential scorer once the threshold reaches this.
        let next_partition_bound = prefix_bounds
            .get(num_non_essential + 1)
            .copied()
            .unwrap_or(Score::INFINITY);
        let (non_essential, essential) = scorers.split_at_mut(num_non_essential);
        docs.clear();
        for scorer in essential.iter_mut() {
            if scorer.doc() < floor {
                scorer.seek(floor);
            }
            docs.push(scorer.doc());
        }
        let mut candidate = docs.iter().copied().min().unwrap_or(TERMINATED);
        // Inner loop: one candidate per iteration, with a single pass over the essential
        // scorers that scores, advances, and finds the next candidate.
        //
        // No essential-bound check is needed before scoring: every essential scorer's bound
        // plus all non-essential bounds already exceeds the threshold, by construction.
        loop {
            if candidate == TERMINATED {
                return;
            }
            let mut score: Score = 0.0;
            let mut next_candidate = TERMINATED;
            let mut exhausted = false;
            for (doc, scorer) in docs.iter_mut().zip(essential.iter_mut()) {
                if *doc == candidate {
                    score += scorer.score();
                    *doc = scorer.advance();
                    exhausted |= *doc == TERMINATED;
                }
                next_candidate = next_candidate.min(*doc);
            }
            let mut is_competitive = true;
            for (ord, scorer) in non_essential.iter_mut().enumerate().rev() {
                if score + prefix_bounds[ord + 1] <= threshold {
                    is_competitive = false;
                    break;
                }
                let mut doc = scorer.doc();
                if doc < candidate {
                    doc = scorer.seek(candidate);
                }
                if doc == candidate {
                    score += scorer.score();
                }
            }
            let mut repartition = exhausted;
            if is_competitive && score > threshold {
                let new_threshold = callback(candidate, score);
                repartition |= new_threshold >= next_partition_bound || new_threshold < threshold;
                threshold = new_threshold;
            }
            floor = candidate + 1;
            candidate = next_candidate;
            if repartition {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::maxscore;
    use crate::query::term_query::TermScorer;
    use crate::query::Bm25Weight;
    use crate::{DocId, Score};

    fn term_scorer(docs: impl Iterator<Item = DocId>, fieldnorms: &[u32]) -> TermScorer {
        let postings: Vec<(DocId, u32)> = docs.map(|doc| (doc, 1)).collect();
        let bm25_weight = Bm25Weight::for_one_term(postings.len() as u64, 1_000, 10.0);
        TermScorer::create_for_test(&postings, fieldnorms, bm25_weight)
    }

    // The collector only raises the threshold. If a callback lowers it, a scorer can become
    // essential again while it is behind already evaluated documents. Documents must still be
    // passed in strictly increasing order.
    #[test]
    fn test_maxscore_threshold_decrease_keeps_doc_order() {
        let fieldnorms = vec![10u32; 1_000];
        let dense = term_scorer(0..1_000, &fieldnorms);
        let low_threshold_bound = dense.max_score();
        let scorers = vec![
            dense,
            term_scorer((0..1_000).step_by(7), &fieldnorms),
            term_scorer((0..1_000).step_by(11), &fieldnorms),
        ];
        let mut docs: Vec<DocId> = Vec::new();
        maxscore(scorers, Score::MIN, &mut |doc, _score| {
            docs.push(doc);
            // First make the dense term non-essential, then lower the threshold again.
            if docs.len() == 1 {
                low_threshold_bound
            } else {
                Score::MIN
            }
        });
        assert!(docs.len() > 2);
        assert!(docs.windows(2).all(|pair| pair[0] < pair[1]), "{docs:?}");
    }
}
