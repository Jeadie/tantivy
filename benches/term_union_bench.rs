// Latency of top-K unions of term queries. These run on the Block-Max WAND executor.
//
// What's measured:
// - `a OR b OR ...` term unions (`BooleanQuery::new_multiterms_query`), top-K by BM25 score.
// - Matrix: query mix x number of terms (2, 3, 5, 10, 20) x K (10, 100, 1000).
// - Each cell times many distinct queries of its shape, drawn from the vocabulary with a seed that
//   depends on the cell. The samples go through the queries in turn, so the percentiles describe
//   the query mix of the cell, not only the timing noise of one query.
// - P90, P95, P99 and P99.9 latency per cell. binggan only reports min/median/average/max, so this
//   bench collects its own samples.
//
// Corpus model (seeded RNG; 1M docs in one segment by default). Each class has a document
// frequency per term, with a fixed spread of 0.5x to 2x between the terms of the class:
// - rare: 200 terms, 0.1% of docs; medium: 100 terms, 3%; dense: 40 terms, 25%;
// - correlated: 100 terms, in 10% of the 5% "topic" docs and 0.1% of the other docs.
// - Term frequencies follow a geometric-like distribution; filler tokens vary field lengths.
//
// Each cell also prints a checksum of the top-K (doc address, score bits) of all its queries, so
// that two builds can be checked for identical results.
//
// Run with: cargo bench --bench term_union_bench
// Environment variables:
// - TERM_UNION_BENCH_FILTER: only run cells whose label (`mix/terms/k`) contains this substring.
// - TERM_UNION_BENCH_INDEX_DIR: build the index in this directory once, then reuse it. Without it,
//   the index is built in RAM on every run.
// - TERM_UNION_BENCH_BUDGET_SECS: time budget per cell after the minimum sample count (default 3).
// - TERM_UNION_BENCH_NUM_DOCS: number of documents (default 1,000,000).
// - TERM_UNION_BENCH_SEGMENTS: number of segments, with equal document counts (default 1). Each
//   segment collects its own top K from a threshold of zero, as in production searches.
// - TERM_UNION_BENCH_MIN_SAMPLES / TERM_UNION_BENCH_MAX_SAMPLES: samples per cell (default 1,000
//   and 10,000). P99 needs at least 1,000 samples to have 10 samples above it, P99.9 needs 10,000.
// - TERM_UNION_BENCH_QUERIES: distinct queries per cell (default 100).

use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::Path;
use std::time::{Duration, Instant};

use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tantivy::collector::TopDocs;
use tantivy::indexer::NoMergePolicy;
use tantivy::query::BooleanQuery;
use tantivy::schema::{Field, Schema, TEXT};
use tantivy::{doc, Index, ReloadPolicy, Searcher, Term};

const DEFAULT_NUM_DOCS: usize = 1_000_000;
const DEFAULT_NUM_QUERIES: usize = 100;
/// Changes when the corpus model changes, so that an index directory of an older model is refused.
const CORPUS_VERSION: u32 = 2;
const NUM_TERMS: [usize; 5] = [2, 3, 5, 10, 20];
const TOP_K: [usize; 3] = [10, 100, 1000];

/// P99 needs at least 10 samples above it.
const DEFAULT_MIN_SAMPLES: usize = 1_000;
/// P99.9 needs at least 10 samples above it.
const DEFAULT_MAX_SAMPLES: usize = 10_000;
/// Percentiles printed for each cell. A percentile q needs at least 10 / (1 - q) samples to have
/// 10 samples above it: 100 for P90, 200 for P95, 1,000 for P99 and 10,000 for P99.9.
const PERCENTILES: [(&str, f64); 4] = [
    ("p90_us", 0.90),
    ("p95_us", 0.95),
    ("p99_us", 0.99),
    ("p999_us", 0.999),
];
const DEFAULT_CELL_BUDGET_SECS: f64 = 3.0;
const WARMUP_RUNS: usize = 20;

/// Settings read from the environment.
struct BenchConfig {
    filter: String,
    cell_budget: Duration,
    num_docs: usize,
    num_segments: usize,
    min_samples: usize,
    max_samples: usize,
    num_queries: usize,
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name} has an invalid value: {value:?}")),
        Err(_) => default,
    }
}

impl BenchConfig {
    fn from_env() -> Self {
        let config = BenchConfig {
            filter: env_or("TERM_UNION_BENCH_FILTER", String::new()),
            cell_budget: Duration::from_secs_f64(env_or(
                "TERM_UNION_BENCH_BUDGET_SECS",
                DEFAULT_CELL_BUDGET_SECS,
            )),
            num_docs: env_or("TERM_UNION_BENCH_NUM_DOCS", DEFAULT_NUM_DOCS),
            num_segments: env_or("TERM_UNION_BENCH_SEGMENTS", 1),
            min_samples: env_or("TERM_UNION_BENCH_MIN_SAMPLES", DEFAULT_MIN_SAMPLES),
            max_samples: env_or("TERM_UNION_BENCH_MAX_SAMPLES", DEFAULT_MAX_SAMPLES),
            num_queries: env_or("TERM_UNION_BENCH_QUERIES", DEFAULT_NUM_QUERIES),
        };
        assert!(config.num_segments >= 1 && config.num_segments <= config.num_docs);
        assert!(config.min_samples >= 1 && config.min_samples <= config.max_samples);
        assert!(config.num_queries >= 1);
        config
    }

    /// Identifies the corpus, so that an index directory is only reused for the same corpus.
    fn corpus_id(&self) -> String {
        format!(
            "version={CORPUS_VERSION} docs={} segments={}",
            self.num_docs, self.num_segments
        )
    }
}

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

    /// Number of terms of the class. There are enough terms to draw many distinct queries of
    /// 20 terms, and few enough dense terms to keep documents short.
    fn vocabulary_size(self) -> usize {
        match self {
            TermClass::Rare => 200,
            TermClass::Medium => 100,
            TermClass::Dense => 40,
            TermClass::Correlated => 100,
        }
    }

    /// Probability that a document contains a given term of the class, before the per-term
    /// spread.
    fn base_probability(self, is_topic_doc: bool) -> f64 {
        match self {
            TermClass::Rare => 0.001,
            TermClass::Medium => 0.03,
            TermClass::Dense => 0.25,
            TermClass::Correlated if is_topic_doc => 0.1,
            TermClass::Correlated => 0.001,
        }
    }

    /// Per-term factors between 0.5 and 2 that spread the document frequencies inside the class.
    fn term_spread(self) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(self as u64 + 1);
        (0..self.vocabulary_size())
            .map(|_| 2f64.powf(rng.random_range(-1.0..=1.0)))
            .collect()
    }

    /// Draws `count` distinct terms of the class.
    fn sample_tokens(self, count: usize, rng: &mut StdRng) -> Vec<String> {
        rand::seq::index::sample(rng, self.vocabulary_size(), count)
            .into_iter()
            .map(|ord| self.token(ord))
            .collect()
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

    /// Draws the tokens of one query with `num_terms` distinct terms.
    fn sample_query_tokens(self, num_terms: usize, rng: &mut StdRng) -> Vec<String> {
        match self {
            QueryMix::Rare => TermClass::Rare.sample_tokens(num_terms, rng),
            QueryMix::Medium => TermClass::Medium.sample_tokens(num_terms, rng),
            QueryMix::Dense => TermClass::Dense.sample_tokens(num_terms, rng),
            QueryMix::Correlated => TermClass::Correlated.sample_tokens(num_terms, rng),
            QueryMix::Skewed => {
                let mut tokens = TermClass::Dense.sample_tokens(1, rng);
                tokens.extend(TermClass::Rare.sample_tokens(num_terms - 1, rng));
                tokens
            }
            // The same class counts as cycling through rare, medium and dense.
            QueryMix::Mixed => {
                let mut tokens = TermClass::Rare.sample_tokens(num_terms.div_ceil(3), rng);
                tokens.extend(TermClass::Medium.sample_tokens((num_terms + 1) / 3, rng));
                tokens.extend(TermClass::Dense.sample_tokens(num_terms / 3, rng));
                tokens
            }
        }
    }

    /// The distinct queries of a cell. They depend only on the mix, the number of terms and the
    /// number of queries, so every binary and every K of the cell uses the same queries.
    fn queries(self, num_terms: usize, num_queries: usize, body: Field) -> Vec<BooleanQuery> {
        let mut hasher = DefaultHasher::new();
        (self.name(), num_terms).hash(&mut hasher);
        let mut rng = StdRng::seed_from_u64(hasher.finish());
        (0..num_queries)
            .map(|_| {
                let terms: Vec<Term> = self
                    .sample_query_tokens(num_terms, &mut rng)
                    .iter()
                    .map(|token| Term::from_field_text(body, token))
                    .collect();
                BooleanQuery::new_multiterms_query(terms)
            })
            .collect()
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

fn fill_index(index: &Index, body: Field, config: &BenchConfig) {
    let mut rng = StdRng::from_seed([7u8; 32]);
    let tokens: Vec<Vec<String>> = TermClass::ALL
        .iter()
        .map(|class| {
            (0..class.vocabulary_size())
                .map(|ord| class.token(ord))
                .collect()
        })
        .collect();
    let spreads: Vec<Vec<f64>> = TermClass::ALL
        .iter()
        .map(|class| class.term_spread())
        .collect();
    let mut writer = index.writer_with_num_threads(1, 2_000_000_000).unwrap();
    writer.set_merge_policy(Box::new(NoMergePolicy));
    let docs_per_segment = config.num_docs.div_ceil(config.num_segments);
    let mut text = String::new();
    for doc_ord in 0..config.num_docs {
        if doc_ord > 0 && doc_ord % docs_per_segment == 0 {
            commit_one_segment(index, &mut writer);
        }
        text.clear();
        let is_topic_doc = rng.random_bool(0.05);
        for ((class, class_tokens), spread) in TermClass::ALL.iter().zip(&tokens).zip(&spreads) {
            let probability = class.base_probability(is_topic_doc);
            for (token, factor) in class_tokens.iter().zip(spread) {
                if rng.random_bool(probability * factor) {
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
    commit_one_segment(index, &mut writer);
    writer.wait_merging_threads().unwrap();
}

/// Commits the pending documents as exactly one new segment. The writer can flush a large commit
/// into several segments when its memory budget is full, so those segments are merged.
fn commit_one_segment(index: &Index, writer: &mut tantivy::IndexWriter) {
    let before = index.searchable_segment_ids().unwrap();
    writer.commit().unwrap();
    let new_segments: Vec<_> = index
        .searchable_segment_ids()
        .unwrap()
        .into_iter()
        .filter(|segment_id| !before.contains(segment_id))
        .collect();
    if new_segments.len() > 1 {
        writer.merge(&new_segments).wait().unwrap();
    }
}

fn open_or_build_index(config: &BenchConfig) -> (Searcher, Field) {
    let mut schema_builder = Schema::builder();
    let body = schema_builder.add_text_field("body", TEXT);
    let schema = schema_builder.build();
    let index = match std::env::var_os("TERM_UNION_BENCH_INDEX_DIR") {
        Some(dir) => {
            let dir = Path::new(&dir);
            // The corpus file is written after the last commit, so its presence means a complete
            // index.
            let corpus_file = dir.join("bench_corpus.txt");
            if corpus_file.try_exists().unwrap() {
                let corpus_id = std::fs::read_to_string(&corpus_file).unwrap();
                assert_eq!(
                    corpus_id,
                    config.corpus_id(),
                    "{} holds an index for another corpus",
                    dir.display()
                );
                Index::open_in_dir(dir).unwrap()
            } else {
                std::fs::create_dir_all(dir).unwrap();
                let index = Index::create_in_dir(dir, schema).unwrap();
                fill_index(&index, body, config);
                std::fs::write(&corpus_file, config.corpus_id()).unwrap();
                index
            }
        }
        None => {
            let index = Index::create_in_ram(schema);
            fill_index(&index, body, config);
            index
        }
    };
    let body = index.schema().get_field("body").unwrap();
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    let searcher = reader.searcher();
    assert_eq!(searcher.segment_readers().len(), config.num_segments);
    (searcher, body)
}

fn percentile(sorted_samples: &[Duration], quantile: f64) -> Duration {
    let rank = (quantile * sorted_samples.len() as f64).ceil() as usize;
    sorted_samples[rank.clamp(1, sorted_samples.len()) - 1]
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

fn search(searcher: &Searcher, query: &BooleanQuery, k: usize) -> Vec<(f32, tantivy::DocAddress)> {
    searcher
        .search(query, &TopDocs::with_limit(k).order_by_score())
        .unwrap()
}

fn checksum(hits: &[(f32, tantivy::DocAddress)]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for (score, address) in hits {
        score.to_bits().hash(&mut hasher);
        address.segment_ord.hash(&mut hasher);
        address.doc_id.hash(&mut hasher);
    }
    hasher.finish()
}

/// Combines the checksums of the queries of a cell, in order.
fn combine_checksums(checksums: impl Iterator<Item = u64>) -> u64 {
    let mut hasher = DefaultHasher::new();
    for checksum in checksums {
        checksum.hash(&mut hasher);
    }
    hasher.finish()
}

fn main() {
    let config = BenchConfig::from_env();
    let build_start = Instant::now();
    let (searcher, body) = open_or_build_index(&config);
    eprintln!(
        "index ready: {} docs, {} segment(s) in {:.1}s",
        searcher.num_docs(),
        searcher.segment_readers().len(),
        build_start.elapsed().as_secs_f64()
    );
    for class in TermClass::ALL {
        let doc_freqs: Vec<u64> = (0..class.vocabulary_size())
            .map(|ord| {
                searcher
                    .doc_freq(&Term::from_field_text(body, &class.token(ord)))
                    .unwrap()
            })
            .collect();
        let percent = |doc_freq: u64| 100.0 * doc_freq as f64 / config.num_docs as f64;
        eprintln!(
            "{:>7}: {} terms, doc_freq {:.2}% to {:.2}%",
            class.prefix(),
            doc_freqs.len(),
            percent(*doc_freqs.iter().min().unwrap()),
            percent(*doc_freqs.iter().max().unwrap()),
        );
    }
    eprintln!("{} distinct queries per cell", config.num_queries);

    let percentile_names: Vec<String> = PERCENTILES
        .iter()
        .map(|(name, _)| format!("{name:>12}"))
        .collect();
    println!(
        "{:<11} {:>5} {:>5} {:>8} {} {:>16}",
        "mix",
        "terms",
        "k",
        "samples",
        percentile_names.join(" "),
        "checksum"
    );
    for mix in QueryMix::ALL {
        for num_terms in NUM_TERMS {
            let queries = mix.queries(num_terms, config.num_queries, body);
            for k in TOP_K {
                let label = format!("{}/{num_terms}/{k}", mix.name());
                if !label.contains(&config.filter) {
                    continue;
                }
                // This also runs every query once before the timed samples.
                let result_checksum = combine_checksums(
                    queries
                        .iter()
                        .map(|query| checksum(&search(&searcher, query, k))),
                );
                for warmup in 0..WARMUP_RUNS {
                    let query = &queries[warmup % queries.len()];
                    std::hint::black_box(search(&searcher, query, k));
                }
                let mut samples: Vec<Duration> = Vec::with_capacity(config.max_samples);
                let cell_start = Instant::now();
                while samples.len() < config.max_samples
                    && (samples.len() < config.min_samples
                        || cell_start.elapsed() < config.cell_budget)
                {
                    let query = &queries[samples.len() % queries.len()];
                    let start = Instant::now();
                    std::hint::black_box(search(&searcher, query, k));
                    samples.push(start.elapsed());
                }
                samples.sort();
                let percentile_values: Vec<String> = PERCENTILES
                    .iter()
                    .map(|&(_, quantile)| {
                        format!("{:>12.1}", micros(percentile(&samples, quantile)))
                    })
                    .collect();
                println!(
                    "{:<11} {:>5} {:>5} {:>8} {} {:>16x}",
                    mix.name(),
                    num_terms,
                    k,
                    samples.len(),
                    percentile_values.join(" "),
                    result_checksum,
                );
                eprintln!("done {label} in {:.1}s", cell_start.elapsed().as_secs_f64());
            }
        }
    }
}
