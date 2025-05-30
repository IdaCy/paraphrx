/*
cargo summary \
  --manifest-path c_assess_inf/Cargo.toml \
  --release \
  -- \
  b_tests/summary
*/

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::Value;
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

/// Command line arguments
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// Directory that contains JSON files to evaluate
    directory: PathBuf,
}

const METRIC_COUNT: usize = 10;

/// Per-paraphrase, per-metric aggregates
#[derive(Clone, Debug)]
struct ParaphraseAgg {
    count: usize,
    sum: [u64; METRIC_COUNT],
    min: [u8; METRIC_COUNT],
    max: [u8; METRIC_COUNT],
}

impl ParaphraseAgg {
    fn new() -> Self {
        Self {
            count: 0,
            sum: [0; METRIC_COUNT],
            min: [u8::MAX; METRIC_COUNT],
            max: [u8::MIN; METRIC_COUNT],
        }
    }

    fn update(&mut self, scores: &[u8]) {
        self.count += 1;
        for (i, &s) in scores.iter().enumerate().take(METRIC_COUNT) {
            self.sum[i] += s as u64;
            self.min[i] = self.min[i].min(s);
            self.max[i] = self.max[i].max(s);
        }
    }

    fn avg(&self, i: usize) -> f64 {
        self.sum[i] as f64 / self.count as f64
    }

    /// Average over *all* metrics (macro-score)
    fn overall_avg(&self) -> f64 {
        let total: u64 = self.sum.iter().take(METRIC_COUNT).sum();
        total as f64 / (self.count * METRIC_COUNT) as f64
    }
}

/// Global, cross-paraphrase variability for each metric
#[derive(Clone, Debug)]
struct MetricAgg {
    min: u8,
    max: u8,
    sum: u64,
    count: u64,
}

impl MetricAgg {
    fn new() -> Self {
        Self {
            min: u8::MAX,
            max: u8::MIN,
            sum: 0,
            count: 0,
        }
    }

    fn update(&mut self, score: u8) {
        self.min = self.min.min(score);
        self.max = self.max.max(score);
        self.sum += score as u64;
        self.count += 1;
    }

    fn avg(&self) -> f64 {
        self.sum as f64 / self.count as f64
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if !cli.directory.is_dir() {
        bail!("{} is not a directory", cli.directory.display());
    }

    let mut by_paraphrase: HashMap<String, ParaphraseAgg> = HashMap::new();
    let mut by_metric: [MetricAgg; METRIC_COUNT] = std::array::from_fn(|_| MetricAgg::new());

    let json_files = fs::read_dir(&cli.directory)
        .with_context(|| format!("Reading {}", cli.directory.display()))?;

    for entry in json_files {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        if entry.path().extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        process_file(&entry.path(), &mut by_paraphrase, &mut by_metric)?;
    }

    if by_paraphrase.is_empty() {
        bail!("No valid JSON files found in {}", cli.directory.display());
    }

    report(&by_paraphrase, &by_metric);
    Ok(())
}

fn process_file(
    path: &Path,
    by_paraphrase: &mut HashMap<String, ParaphraseAgg>,
    by_metric: &mut [MetricAgg; METRIC_COUNT],
) -> Result<()> {
    let data = fs::read_to_string(path).with_context(|| format!("Reading {}", path.display()))?;
    let records: Vec<Value> =
        serde_json::from_str(&data).with_context(|| format!("Parsing {}", path.display()))?;

    for rec in records {
        let obj = rec
            .as_object()
            .with_context(|| format!("Top level JSON value must be object in {}", path.display()))?;

        for (key, val) in obj {
            if key == "prompt_count" || key == "prompt_id" {
                continue;
            }
            let arr = val
                .as_array()
                .with_context(|| format!("Field {key} is not an array in {}", path.display()))?;
            if arr.len() != METRIC_COUNT {
                bail!(
                    "{key} array length is {}, expected {METRIC_COUNT} in {}",
                    arr.len(),
                    path.display()
                );
            }
            let scores: Vec<u8> = arr
                .iter()
                .map(|v| {
                    v.as_u64()
                        .with_context(|| format!("Non-integer score in {key} of {}", path.display()))
                        .map(|n| n as u8)
                })
                .collect::<Result<_>>()?;

            let entry = by_paraphrase.entry(key.clone()).or_insert_with(ParaphraseAgg::new);
            entry.update(&scores);

            // update global variability stats
            for (i, &s) in scores.iter().enumerate() {
                by_metric[i].update(s);
            }
        }
    }
    Ok(())
}

fn report(by_para: &HashMap<String, ParaphraseAgg>, by_metric: &[MetricAgg; METRIC_COUNT]) {
    println!("\n================== PARAPHRASE STATS ==================");
    for (p, stats) in by_para {
        println!("► {p}");
        for i in 0..METRIC_COUNT {
            println!(
                "    Metric {:2}:  avg {:4.2}   min {}   max {}",
                i + 1,
                stats.avg(i),
                stats.min[i],
                stats.max[i]
            );
        }
        println!(
            "    → overall average across all metrics: {:.3}\n",
            stats.overall_avg()
        );
    }

    println!("================== TOP-3 BY EACH METRIC ==================");
    for m in 0..METRIC_COUNT {
        let mut v: Vec<(&str, f64)> = by_para
            .iter()
            .map(|(name, s)| (name.as_str(), s.avg(m)))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("Metric {:2}:", m + 1);
        for (rank, (name, score)) in v.into_iter().take(3).enumerate() {
            println!("    #{rank}: {name}   ({:.2})", score);
        }
    }

    println!("\n================== TOP-3 OVERALL ==================");
    let mut overall: Vec<(&str, f64)> = by_para
        .iter()
        .map(|(n, s)| (n.as_str(), s.overall_avg()))
        .collect();
    overall.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (rank, (name, score)) in overall.into_iter().take(3).enumerate() {
        println!("#{rank}: {name}   ({:.3})", score);
    }

    println!("\n================== METRIC VARIABILITY ==================");
    for (i, agg) in by_metric.iter().enumerate() {
        println!(
            "Metric {:2}: min {}   max {}   avg {:4.2}   {}",
            i + 1,
            agg.min,
            agg.max,
            agg.avg(),
            if agg.min == agg.max {
                "⚠ no variability"
            } else {
                ""
            }
        );
    }
}
