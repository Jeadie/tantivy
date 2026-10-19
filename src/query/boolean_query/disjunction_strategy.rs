use std::num::NonZeroUsize;

/// Executor used to collect the top documents of an additive union of term queries
/// (e.g. `a OR b OR c` scored with BM25) in one segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisjunctionStrategy {
    /// Block-Max WAND.
    BlockWand,
    /// MaxScore.
    MaxScore,
}

/// Fixed rule that selects a [`DisjunctionStrategy`] for each segment.
///
/// The rule only applies to unions of term queries whose scores are summed, when the
/// collector prunes by score (e.g. [`TopDocs`](crate::collector::TopDocs) ordered by score).
/// Other queries and collectors are not affected.
///
/// Set it for all queries of a reader with
/// [`IndexReaderBuilder::disjunction_strategy_config`](crate::IndexReaderBuilder::disjunction_strategy_config),
/// or for one query with
/// [`BooleanQuery::with_disjunction_strategy_config`](crate::query::BooleanQuery::with_disjunction_strategy_config).
///
/// The default always selects [`DisjunctionStrategy::BlockWand`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DisjunctionStrategyConfig {
    /// `None` always selects Block-Max WAND.
    max_score_min_terms: Option<NonZeroUsize>,
}

impl DisjunctionStrategyConfig {
    /// Always use Block-Max WAND.
    pub const fn block_wand() -> Self {
        DisjunctionStrategyConfig {
            max_score_min_terms: None,
        }
    }

    /// Use MaxScore when the union has at least `min_terms` terms with postings in the
    /// segment, and Block-Max WAND otherwise.
    ///
    /// A union with a single remaining term always uses a dedicated single-term path,
    /// so `min_terms` values of 1 and 2 behave the same.
    pub const fn max_score_from_num_terms(min_terms: NonZeroUsize) -> Self {
        DisjunctionStrategyConfig {
            max_score_min_terms: Some(min_terms),
        }
    }

    /// The term count at which MaxScore is selected, if any.
    pub fn max_score_min_terms(&self) -> Option<NonZeroUsize> {
        self.max_score_min_terms
    }

    /// Selects the executor for a union with `num_terms` terms that have postings in the
    /// segment.
    pub fn select(&self, num_terms: usize) -> DisjunctionStrategy {
        match self.max_score_min_terms {
            Some(min_terms) if num_terms >= min_terms.get() => DisjunctionStrategy::MaxScore,
            _ => DisjunctionStrategy::BlockWand,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{DisjunctionStrategy, DisjunctionStrategyConfig};

    #[test]
    fn test_default_is_block_wand() {
        let config = DisjunctionStrategyConfig::default();
        assert_eq!(config, DisjunctionStrategyConfig::block_wand());
        for num_terms in 0..64 {
            assert_eq!(config.select(num_terms), DisjunctionStrategy::BlockWand);
        }
    }

    #[test]
    fn test_max_score_from_num_terms() {
        let config =
            DisjunctionStrategyConfig::max_score_from_num_terms(NonZeroUsize::new(4).unwrap());
        assert_eq!(config.select(0), DisjunctionStrategy::BlockWand);
        assert_eq!(config.select(3), DisjunctionStrategy::BlockWand);
        assert_eq!(config.select(4), DisjunctionStrategy::MaxScore);
        assert_eq!(config.select(5), DisjunctionStrategy::MaxScore);
    }
}
