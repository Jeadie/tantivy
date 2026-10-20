// Latency of the Block-Max WAND executor for top-K unions of term queries.
//
// What's measured:
// - `a OR b OR ...` term unions (`BooleanQuery::new_multiterms_query`), top-K by BM25 score.
// - Matrix: query mix x number of terms (2, 3, 5, 10, 20) x K (10, 100, 1000).
// - P99 and P99.9 latency per cell. binggan only reports min/median/average/max, so this bench
//   collects its own samples.
//
// Corpus model (1M docs, single segment, seeded RNG):
// - 20 terms in each of four classes: rare (0.1% of docs), medium (3%), dense (25%), and
//   correlated (50% of the 5% "topic" docs, 0.5% of the others).
// - Term frequencies follow a geometric-like distribution; filler tokens vary field lengths.
//
// Each cell also prints a checksum of the top-K (doc address, score bits), so that two builds
// can be checked for identical results.
//
// Run with: cargo bench --bench block_wand_union_bench
// Environment variables:
// - BLOCK_WAND_BENCH_FILTER: only run cells whose label (`mix/terms/k`) contains this substring.
// - BLOCK_WAND_BENCH_INDEX_DIR: build the index in this directory once, then reuse it. Without
//   it, the index is built in RAM on every run.
// - BLOCK_WAND_BENCH_BUDGET_SECS: time budget per cell after the minimum sample count (default 3).

use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::Path;
use std::time::{Duration, Instant};

use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tantivy::collector::TopDocs;
use tantivy::query::BooleanQuery;
use tantivy::schema::{Field, Schema, TEXT};
use tantivy::{doc, Index, ReloadPolicy, Searcher, Term};

const NUM_DOCS: usize = 1_000_000;
const TERMS_PER_CLASS: usize = 20;
const NUM_TERMS: [usize; 5] = [2, 3, 5, 10, 20];
const TOP_K: [usize; 3] = [10, 100, 1000];

/// P99 needs at least 10 samples above it.
const MIN_SAMPLES: usize = 1_000;
/// P99.9 needs at least 10 samples above it.
const P999_MIN_SAMPLES: usize = 10_000;
const MAX_SAMPLES: usize = 10_000;
const DEFAULT_CELL_BUDGET_SECS: f64 = 3.0;
const WARMUP_RUNS: usize = 20;

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

fn fill_index(index: &Index, body: Field) {
    let mut rng = StdRng::from_seed([7u8; 32]);
    let tokens: Vec<Vec<String>> = TermClass::ALL
        .iter()
        .map(|class| (0..TERMS_PER_CLASS).map(|ord| class.token(ord)).collect())
        .collect();
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

fn open_or_build_index() -> (Searcher, Field) {
    let mut schema_builder = Schema::builder();
    let body = schema_builder.add_text_field("body", TEXT);
    let schema = schema_builder.build();
    let index = match std::env::var_os("BLOCK_WAND_BENCH_INDEX_DIR") {
        Some(dir) => {
            let dir = Path::new(&dir);
            // `meta.json` is written by the commit, so its presence means a complete index.
            if dir.join("meta.json").try_exists().unwrap() {
                Index::open_in_dir(dir).unwrap()
            } else {
                std::fs::create_dir_all(dir).unwrap();
                let index = Index::create_in_dir(dir, schema).unwrap();
                fill_index(&index, body);
                index
            }
        }
        None => {
            let index = Index::create_in_ram(schema);
            fill_index(&index, body);
            index
        }
    };
    let body = index.schema().get_field("body").unwrap();
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::Manual)
        .try_into()
        .unwrap();
    (reader.searcher(), body)
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

fn main() {
    let filter = std::env::var("BLOCK_WAND_BENCH_FILTER").unwrap_or_default();
    let cell_budget = Duration::from_secs_f64(
        std::env::var("BLOCK_WAND_BENCH_BUDGET_SECS")
            .ok()
            .map(|secs| secs.parse().unwrap())
            .unwrap_or(DEFAULT_CELL_BUDGET_SECS),
    );
    let build_start = Instant::now();
    let (searcher, body) = open_or_build_index();
    eprintln!(
        "index ready: {} docs, {} segment(s) in {:.1}s",
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

    println!(
        "{:<11} {:>5} {:>5} {:>8} {:>12} {:>12} {:>9} {:>16}",
        "mix", "terms", "k", "samples", "p99_us", "p999_us", "p999_ok", "checksum"
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
                let result_checksum = checksum(&search(&searcher, &query, k));
                for _ in 0..WARMUP_RUNS {
                    std::hint::black_box(search(&searcher, &query, k));
                }
                let mut samples: Vec<Duration> = Vec::with_capacity(MAX_SAMPLES);
                let cell_start = Instant::now();
                while samples.len() < MAX_SAMPLES
                    && (samples.len() < MIN_SAMPLES || cell_start.elapsed() < cell_budget)
                {
                    let start = Instant::now();
                    std::hint::black_box(search(&searcher, &query, k));
                    samples.push(start.elapsed());
                }
                samples.sort();
                println!(
                    "{:<11} {:>5} {:>5} {:>8} {:>12.1} {:>12.1} {:>9} {:>16x}",
                    mix.name(),
                    num_terms,
                    k,
                    samples.len(),
                    micros(percentile(&samples, 0.99)),
                    micros(percentile(&samples, 0.999)),
                    samples.len() >= P999_MIN_SAMPLES,
                    result_checksum,
                );
                eprintln!("done {label} in {:.1}s", cell_start.elapsed().as_secs_f64());
            }
        }
    }
}
