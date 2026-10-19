// Compares the executors for top-K unions of term queries: Block-Max WAND and MaxScore.
//
// What's measured:
// - `a OR b OR ...` term unions, top-K by BM25 score, with each executor forced through
//   `DisjunctionStrategyConfig`.
// - Matrix: query mix x number of terms (2, 3, 5, 10, 20) x K (10, 100, 1000).
// - P99 and P99.9 latency of each executor, from interleaved runs on the same searcher.
//
// Corpus model (1M docs, single segment, seeded RNG):
// - 20 terms in each of four classes: rare (0.1% of docs), medium (3%), dense (25%), and
//   correlated (50% of the 5% "topic" docs, 0.5% of the others).
// - Term frequencies follow a geometric-like distribution; filler tokens vary field lengths.
//
// binggan only reports min/median/average/max, so this bench collects its own samples.
//
// Run with: cargo bench --bench disjunction_strategy
// Filter cells with: DISJUNCTION_BENCH_FILTER=<substring of the cell label>

use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, DisjunctionStrategyConfig};
use tantivy::schema::{Field, Schema, TEXT};
use tantivy::{doc, Index, ReloadPolicy, Searcher, Term};

const NUM_DOCS: usize = 1_000_000;
const TERMS_PER_CLASS: usize = 20;
const NUM_TERMS: [usize; 5] = [2, 3, 5, 10, 20];
const TOP_K: [usize; 3] = [10, 100, 1000];

const MIN_SAMPLES: usize = 300;
const MAX_SAMPLES: usize = 3000;
const CELL_BUDGET: Duration = Duration::from_secs(2);
const WARMUP_RUNS: usize = 3;

#[derive(Clone, Copy)]
enum TermClass {
    Rare,
    Medium,
    Dense,
    Correlated,
}

impl TermClass {
    const ALL: [TermClass; 4] = [
        TermClass::Rare,
        TermClass::Medium,
        TermClass::Dense,
        TermClass::Correlated,
    ];

    fn prefix(self) -> &'static str {
        match self {
            TermClass::Rare => "rare",
            TermClass::Medium => "medium",
            TermClass::Dense => "dense",
            TermClass::Correlated => "corr",
        }
    }

    fn token(self, ord: usize) -> String {
        format!("{}{ord}", self.prefix())
    }
}

#[derive(Clone, Copy)]
enum QueryMix {
    Rare,
    Medium,
    Dense,
    /// One dense term, the other terms rare.
    Skewed,
    /// Terms cycle through rare, medium, and dense.
    Mixed,
    /// Terms that tend to occur in the same documents.
    Correlated,
}

impl QueryMix {
    const ALL: [QueryMix; 6] = [
        QueryMix::Rare,
        QueryMix::Medium,
        QueryMix::Dense,
        QueryMix::Skewed,
        QueryMix::Mixed,
        QueryMix::Correlated,
    ];

    fn name(self) -> &'static str {
        match self {
            QueryMix::Rare => "rare",
            QueryMix::Medium => "medium",
            QueryMix::Dense => "dense",
            QueryMix::Skewed => "skewed",
            QueryMix::Mixed => "mixed",
            QueryMix::Correlated => "correlated",
        }
    }

    fn tokens(self, num_terms: usize) -> Vec<String> {
        let single_class = |class: TermClass| (0..num_terms).map(|ord| class.token(ord)).collect();
        match self {
            QueryMix::Rare => single_class(TermClass::Rare),
            QueryMix::Medium => single_class(TermClass::Medium),
            QueryMix::Dense => single_class(TermClass::Dense),
            QueryMix::Correlated => single_class(TermClass::Correlated),
            QueryMix::Skewed => std::iter::once(TermClass::Dense.token(0))
                .chain((0..num_terms - 1).map(|ord| TermClass::Rare.token(ord)))
                .collect(),
            QueryMix::Mixed => (0..num_terms)
                .map(|ord| {
                    let class = [TermClass::Rare, TermClass::Medium, TermClass::Dense][ord % 3];
                    class.token(ord / 3)
                })
                .collect(),
        }
    }
}

/// Generate term frequency from a geometric-like distribution.
/// Most values are 1, a few are 2-3, rarely higher.
fn random_term_freq(rng: &mut StdRng) -> u32 {
    let mut tf = 1u32;
    while tf < 10 && rng.random_bool(0.4) {
        tf += 1;
    }
    tf
}

fn build_index() -> (Searcher, Field) {
    let mut schema_builder = Schema::builder();
    let body = schema_builder.add_text_field("body", TEXT);
    let index = Index::create_in_ram(schema_builder.build());
    let mut rng = StdRng::from_seed([7u8; 32]);
    let tokens: Vec<Vec<String>> = TermClass::ALL
        .iter()
        .map(|class| (0..TERMS_PER_CLASS).map(|ord| class.token(ord)).collect())
        .collect();
    {
        let mut writer = index.writer_with_num_threads(1, 2_000_000_000).unwrap();
        let mut text = String::new();
        for _ in 0..NUM_DOCS {
            text.clear();
            let is_topic_doc = rng.random_bool(0.05);
            for (class, class_tokens) in TermClass::ALL.iter().zip(&tokens) {
                let probability = match class {
                    TermClass::Rare => 0.001,
                    TermClass::Medium => 0.03,
                    TermClass::Dense => 0.25,
                    TermClass::Correlated if is_topic_doc => 0.5,
                    TermClass::Correlated => 0.005,
                };
                for token in class_tokens {
                    if rng.random_bool(probability) {
                        for _ in 0..random_term_freq(&mut rng) {
                            text.push_str(token);
                            text.push(' ');
                        }
                    }
                }
            }
            for _ in 0..rng.random_range(1..30) {
                text.push_str("z ");
            }
            writer.add_document(doc!(body => text.as_str())).unwrap();
        }
        writer.commit().unwrap();
        writer.wait_merging_threads().unwrap();
    }
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    let searcher = reader.searcher();
    (searcher, body)
}

fn percentile(sorted_samples: &[Duration], quantile: f64) -> Duration {
    let rank = (quantile * sorted_samples.len() as f64).ceil() as usize;
    sorted_samples[rank.clamp(1, sorted_samples.len()) - 1]
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

struct ExecutorRun {
    query: BooleanQuery,
    samples: Vec<Duration>,
}

impl ExecutorRun {
    fn new(query: &BooleanQuery, config: DisjunctionStrategyConfig) -> Self {
        ExecutorRun {
            query: query.clone().with_disjunction_strategy_config(config),
            samples: Vec::with_capacity(MAX_SAMPLES),
        }
    }

    fn search(&self, searcher: &Searcher, k: usize) -> Vec<(f32, tantivy::DocAddress)> {
        searcher
            .search(&self.query, &TopDocs::with_limit(k).order_by_score())
            .unwrap()
    }

    fn sample(&mut self, searcher: &Searcher, k: usize) {
        let start = Instant::now();
        std::hint::black_box(self.search(searcher, k));
        self.samples.push(start.elapsed());
    }
}

fn main() {
    let filter = std::env::var("DISJUNCTION_BENCH_FILTER").unwrap_or_default();
    let build_start = Instant::now();
    let (searcher, body) = build_index();
    eprintln!(
        "built index: {} docs, {} segment(s) in {:.1}s",
        searcher.num_docs(),
        searcher.segment_readers().len(),
        build_start.elapsed().as_secs_f64()
    );
    for class in TermClass::ALL {
        let doc_freq = searcher
            .doc_freq(&Term::from_field_text(body, &class.token(0)))
            .unwrap();
        eprintln!(
            "{:>7} terms: doc_freq ~ {doc_freq} ({:.2}%)",
            class.prefix(),
            100.0 * doc_freq as f64 / NUM_DOCS as f64
        );
    }

    let block_wand = DisjunctionStrategyConfig::block_wand();
    let max_score = DisjunctionStrategyConfig::max_score_from_num_terms(NonZeroUsize::MIN);

    println!(
        "{:<11} {:>5} {:>5} {:>8} | {:>12} {:>12} | {:>12} {:>12} | {:>9} {:>9}",
        "mix",
        "terms",
        "k",
        "samples",
        "wand_p99_us",
        "wand_p999_us",
        "maxs_p99_us",
        "maxs_p999_us",
        "p99_ratio",
        "p999_ratio"
    );
    for mix in QueryMix::ALL {
        for num_terms in NUM_TERMS {
            let terms: Vec<Term> = mix
                .tokens(num_terms)
                .iter()
                .map(|token| Term::from_field_text(body, token))
                .collect();
            let query = BooleanQuery::new_multiterms_query(terms);
            for k in TOP_K {
                let label = format!("{}/{num_terms}/{k}", mix.name());
                if !label.contains(&filter) {
                    continue;
                }
                let mut runs = [
                    ExecutorRun::new(&query, block_wand),
                    ExecutorRun::new(&query, max_score),
                ];

                // Both executors must return the same number of hits and the same top score.
                let [wand_hits, max_score_hits] =
                    [&runs[0], &runs[1]].map(|run| run.search(&searcher, k));
                assert_eq!(wand_hits.len(), max_score_hits.len(), "{label}");
                if let (Some(wand_top), Some(max_score_top)) =
                    (wand_hits.first(), max_score_hits.first())
                {
                    assert!(
                        (wand_top.0 - max_score_top.0).abs() <= 1e-4 * wand_top.0,
                        "{label}"
                    );
                }

                for _ in 0..WARMUP_RUNS {
                    for run in &mut runs {
                        run.search(&searcher, k);
                    }
                }
                // Interleave the executors so that drift affects both equally.
                let cell_start = Instant::now();
                while runs[0].samples.len() < MAX_SAMPLES
                    && (runs[0].samples.len() < MIN_SAMPLES || cell_start.elapsed() < CELL_BUDGET)
                {
                    for run in &mut runs {
                        run.sample(&searcher, k);
                    }
                }

                let [wand, max_score] = runs.map(|mut run| {
                    run.samples.sort();
                    (
                        percentile(&run.samples, 0.99),
                        percentile(&run.samples, 0.999),
                        run.samples.len(),
                    )
                });
                println!(
                    "{:<11} {:>5} {:>5} {:>8} | {:>12.1} {:>12.1} | {:>12.1} {:>12.1} | {:>9.2} \
                     {:>9.2}",
                    mix.name(),
                    num_terms,
                    k,
                    wand.2,
                    micros(wand.0),
                    micros(wand.1),
                    micros(max_score.0),
                    micros(max_score.1),
                    micros(max_score.0) / micros(wand.0),
                    micros(max_score.1) / micros(wand.1),
                );
            }
        }
    }
}
