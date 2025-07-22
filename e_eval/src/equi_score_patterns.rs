/*
cargo equi_score_patterns \
    a_data/alpaca/equi_scores/paraphrases_500_part1_scores.json
*/

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use clap::Parser;
use serde::{Deserialize, Serialize};

// Simple stats for one paraphrase type
#[derive(Serialize)]
struct ParaphraseStats {
    count: usize,
    mean: f64,
    distribution: HashMap<u64, usize>,
    median: f64,
}

// Command‐line arguments: just the input JSON file
#[derive(Parser)]
struct Args {
    // Path to the input JSON (array of { prompt_count, scores: {...} })
    input: String,
}

// Each record in the input file
#[derive(Deserialize)]
struct Record {
    // we only need `scores`; `prompt_count` is not used for grouping here
    #[serde(skip)]
    prompt_count: u64,

    scores: HashMap<String, u64>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Open & parse the input JSON
    let file = File::open(&args.input)?;
    let reader = BufReader::new(file);
    let records: Vec<Record> = serde_json::from_reader(reader)?;

    // Group scores by paraphrase key
    let mut buckets: HashMap<String, Vec<u64>> = HashMap::new();

    for rec in records {
        for (para, &score) in rec.scores.iter() {
            buckets.entry(para.clone()).or_default().push(score);
        }
    }

    // Compute stats
    let mut stats_map: HashMap<String, ParaphraseStats> = HashMap::new();
    let mut mean_map: HashMap<String, f64> = HashMap::new();
    let mut median_map: HashMap<String, f64> = HashMap::new();

    for (para, mut scores) in buckets {
        scores.sort_unstable();
        let count = scores.len();
        let sum: u64 = scores.iter().sum();
        let mean = sum as f64 / count as f64;

        // distribution
        let mut dist = HashMap::new();
        for &s in &scores {
            *dist.entry(s).or_default() += 1;
        }

        // median
        let median = if count % 2 == 1 {
            scores[count / 2] as f64
        } else {
            let hi = scores[count / 2];
            let lo = scores[count / 2 - 1];
            (hi as f64 + lo as f64) / 2.0
        };

        stats_map.insert(
            para.clone(),
            ParaphraseStats {
                count,
                mean,
                distribution: dist,
                median,
            },
        );
        mean_map.insert(para.clone(), mean);
        median_map.insert(para, median);
    }

    // Prepare output paths
    let path = Path::new(&args.input);
    let dir  = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("output");

    let stats_path  = dir.join(format!("{stem}_stats.json"));
    let mean_path   = dir.join(format!("{stem}_mean_equi_scores.json"));
    let median_path = dir.join(format!("{stem}_median_equi_scores.json"));

    // Write them out
    {
        let f = File::create(stats_path)?;
        serde_json::to_writer_pretty(f, &stats_map)?;
    }
    {
        let f = File::create(mean_path)?;
        serde_json::to_writer_pretty(f, &mean_map)?;
    }
    {
        let f = File::create(median_path)?;
        serde_json::to_writer_pretty(f, &median_map)?;
    }

    println!("Wrote stats, mean & median JSON files next to `{}`", &args.input);
    Ok(())
}
