use super::block_wand_union::{block_wand, block_wand_single_scorer};
use crate::query::term_query::TermScorer;
use crate::query::Scorer;
use crate::{DocId, DocSet, Score, TERMINATED};

/// Term scorers ordered by increasing term-wide score bound.
struct BoundedScorers {
    scorers: Vec<TermScorer>,
    /// `bounds[i]` is the term-wide score bound of `scorers[i]`.
    bounds: Vec<Score>,
    /// `prefix_bounds[i]` is the sum of `bounds[..i]`. It has `scorers.len() + 1` entries.
    prefix_bounds: Vec<Score>,
}

impl BoundedScorers {
    fn new(mut scorers: Vec<TermScorer>) -> Self {
        scorers.sort_by(|left, right| left.max_score().total_cmp(&right.max_score()));
        let mut bounded_scorers = BoundedScorers {
            scorers,
            bounds: Vec::new(),
            prefix_bounds: Vec::new(),
        };
        bounded_scorers.update_bounds();
        bounded_scorers
    }

    fn update_bounds(&mut self) {
        self.bounds = self.scorers.iter().map(TermScorer::max_score).collect();
        self.prefix_bounds.clear();
        let mut sum = 0.0;
        self.prefix_bounds.push(sum);
        for &bound in &self.bounds {
            sum += bound;
            self.prefix_bounds.push(sum);
        }
    }

    fn remove_exhausted(&mut self) {
        if self.scorers.iter().any(|scorer| scorer.doc() == TERMINATED) {
            self.scorers.retain(|scorer| scorer.doc() != TERMINATED);
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
    loop {
        if bounded_scorers.scorers.len() == 1 {
            let mut scorer = bounded_scorers.scorers.pop().unwrap();
            if scorer.doc() < floor {
                scorer.seek(floor);
            }
            block_wand_single_scorer(scorer, threshold, callback);
            return;
        }
        let num_non_essential = bounded_scorers.num_non_essential(threshold);
        let BoundedScorers {
            scorers,
            bounds,
            prefix_bounds,
        } = &mut bounded_scorers;
        let (non_essential, essential) = scorers.split_at_mut(num_non_essential);
        let essential_bounds = &bounds[num_non_essential..];

        // A scorer that just became essential can be behind `floor`.
        let mut candidate = TERMINATED;
        for scorer in essential.iter_mut() {
            if scorer.doc() < floor {
                scorer.seek(floor);
            }
            candidate = candidate.min(scorer.doc());
        }
        if candidate == TERMINATED {
            return;
        }

        let candidate_essential_bound: Score = essential
            .iter()
            .zip(essential_bounds)
            .filter(|(scorer, _)| scorer.doc() == candidate)
            .map(|(_, bound)| bound)
            .sum();
        if candidate_essential_bound + prefix_bounds[num_non_essential] > threshold {
            let mut score: Score = essential
                .iter_mut()
                .filter(|scorer| scorer.doc() == candidate)
                .map(|scorer| scorer.score())
                .sum();
            let mut is_competitive = true;
            for (ord, scorer) in non_essential.iter_mut().enumerate().rev() {
                if score + prefix_bounds[ord + 1] <= threshold {
                    is_competitive = false;
                    break;
                }
                if scorer.doc() < candidate {
                    scorer.seek(candidate);
                }
                if scorer.doc() == candidate {
                    score += scorer.score();
                }
            }
            if is_competitive && score > threshold {
                threshold = callback(candidate, score);
            }
        }

        for scorer in essential.iter_mut() {
            if scorer.doc() == candidate {
                scorer.advance();
            }
        }
        floor = candidate + 1;
        bounded_scorers.remove_exhausted();
        if bounded_scorers.scorers.is_empty() {
            return;
        }
    }
}
