/*
cargo equi_patterns \
    -o a_data/alpaca/equi_scores/patterns \
    a_data/alpaca/equi_scores/paraphrases_500_part1_scores.json \
    a_data/alpaca/equi_scores/paraphrases_500_part2_scores.json
*/

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use clap::Parser;
use serde::{Deserialize, Serialize};
use anyhow::{Context};

// stats for one paraphrase type
#[derive(Serialize)]
struct ParaphraseStats {
    count: usize,
    mean: f64,
    distribution: HashMap<u64, usize>,
    median: f64,
}

// Command‐line arguments
#[derive(Parser)]
struct Args {
    // One or more input JSON files (arrays of '{ prompt_count, scores: { ... } }')
    #[arg(required = true)]
    input: Vec<PathBuf>,

    // Output prefix: the files written will be
    // '${OUTPUT_PREFIX}_stats.json',
    // '${OUTPUT_PREFIX}_mean_equi_scores.json',
    // '${OUTPUT_PREFIX}_median_equi_scores.json'
    #[arg(short = 'o', long = "output-prefix")]
    output_prefix: PathBuf,
}

// One record in the input JSON
#[derive(Deserialize)]
struct Record {
    // We only need to consume it so it doesn't trigger "unused"
    #[serde(rename = "prompt_count")]
    _prompt_count: u64,

    scores: HashMap<String, u64>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Read & merge all records
    let mut all_records = Vec::new();
    for input_path in &args.input {
        let f = File::open(input_path)
            .with_context(|| format!("cannot open `{}`", input_path.display()))?;
        let rdr = BufReader::new(f);
        let mut recs: Vec<Record> = serde_json::from_reader(rdr)
            .with_context(|| format!("cannot parse JSON in `{}`", input_path.display()))?;
        all_records.append(&mut recs);
    }

    // Bucket scores by paraphrase key
    let mut buckets: HashMap<String, Vec<u64>> = HashMap::new();
    for rec in all_records {
        for (para, &score) in &rec.scores {
            buckets.entry(para.clone()).or_default().push(score);
        }
    }

    // Compute full stats + separate mean/median maps
    let mut stats_map: BTreeMap<String, ParaphraseStats> = BTreeMap::new();
    let mut mean_map: BTreeMap<String, f64> = BTreeMap::new();
    let mut median_map: BTreeMap<String, f64> = BTreeMap::new();

    for (para, mut scores) in buckets {
        scores.sort_unstable();
        let count = scores.len();
        let sum: u64 = scores.iter().sum();
        let mean = sum as f64 / count as f64;

        let mut dist = HashMap::new();
        for &s in &scores {
            *dist.entry(s).or_default() += 1;
        }

        let median = if count % 2 == 1 {
            scores[count / 2] as f64
        } else {
            let hi = scores[count / 2];
            let lo = scores[count / 2 - 1];
            (hi as f64 + lo as f64) / 2.0
        };

        stats_map.insert(
            para.clone(),
            ParaphraseStats { count, mean, distribution: dist, median },
        );
        mean_map.insert(para.clone(), mean);
        median_map.insert(para, median);
    }

    // Write outputs
    let pfx = &args.output_prefix;
    let stats_path  = pfx.with_extension("stats.json");
    let mean_path   = pfx.with_extension("mean_equi_scores.json");
    let median_path = pfx.with_extension("median_equi_scores.json");

    {
        let f = File::create(&stats_path)?;
        serde_json::to_writer_pretty(f, &stats_map)?;
    }
    {
        let f = File::create(&mean_path)?;
        serde_json::to_writer_pretty(f, &mean_map)?;
    }
    {
        let f = File::create(&median_path)?;
        serde_json::to_writer_pretty(f, &median_map)?;
    }

    println!(
        "Wrote:\n • {}\n • {}\n • {}",
        stats_path.display(),
        mean_path.display(),
        median_path.display()
    );
    Ok(())
}
