/*
cargo check_scores \
    --scores-file c_assess_inf/output/alpaca_answer_scores/gemma-2-2b-it.json \
    --max-samples 5
*/

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

// CLI parameters
#[derive(Parser, Debug)]
#[command(version, about = "Summarise Task-Fulfilment scores per prompt")]
struct Args {
    // Path to the scores JSON file
    #[arg(long)]
    scores_file: PathBuf,

    // Process at most this many prompts (omit to process all)
    #[arg(long)]
    max_samples: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // load
    let file = File::open(&args.scores_file)
        .with_context(|| "Cannot open scores file")?;
    let rows: Vec<Value> =
        serde_json::from_reader(file).with_context(|| "Scores JSON malformed")?;

    // hist[prompt_count][score] = count
    let mut hist: HashMap<u64, [u32; 11]> = HashMap::new();

    let mut processed = 0usize;
    for obj in rows {
        if let Some(cap) = args.max_samples {
            if processed >= cap {
                break;
            }
        }

        let prompt_count = obj
            .get("prompt_count")
            .and_then(|v| v.as_u64())
            .context("Every row must have prompt_count")?;

        for (key, val) in obj
            .as_object()
            .expect("Row must be a JSON object")
        {
            if !(key == "instruction_original" || key.starts_with("instruct_")) {
                continue;
            }
            // first metric only
            if let Some(score) = val
                .as_array()
                .and_then(|a| a.first())
                .and_then(|v| v.as_u64())
            {
                if score <= 10 {
                    hist.entry(prompt_count)
                        .or_insert([0u32; 11])[score as usize] += 1;
                }
            }
        }
        processed += 1;
    }

    // print
    println!("{:<6} {:<5} {:<5}", "prompt", "score", "count");
    println!("{:-<6} {:-<5} {:-<5}", "", "", "");

    let mut keys: Vec<_> = hist.keys().copied().collect();
    keys.sort();
    for pc in keys {
        let counts = &hist[&pc];
        for s in (0..=10).rev() {
            let c = counts[s];
            if c > 0 {
                println!("{:<6} {:<5} {:<5}", pc, s, c);
            }
        }
    }

    Ok(())
}
