/*
use:

# compact output, blank-output rows dropped
cargo run --manifest-path a_data/preproc/alpaca_sanity/Cargo.toml -- \
  a_data/alpaca/alpaca_52k_clean.jsonl \
  a_data/alpaca/alpaca_52k_clean_proc.jsonl

# pretty-printed output and KEEP the 28 rows whose `output` is empty
cargo run --release --manifest-path a_data/preproc/alpaca_sanity/Cargo.toml -- \
  a_data/alpaca/alpaca_52k_clean.jsonl \
  a_data/alpaca/alpaca_52k_clean_proc_pretty.jsonl \
  --pretty --keep-empty-output
*/

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader, Write},
    path::PathBuf,
};
use uuid::Uuid;

#[derive(Debug, Deserialize, Serialize)]
struct Prompt {
    #[serde(default)]
    prompt_id: Option<String>,
    #[serde(default)]
    prompt_count: Option<u64>, // added sequential id
    instruction: String,
    #[serde(default)]
    input: String,
    #[serde(default)]
    output: String,
}

fn main() -> Result<()> {
    // CLI
    let mut args = std::env::args().skip(1);
    let in_path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("pass path to JSONL as first argument"))?;
    let out_path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("pass output path as second argument"))?;
    let mut pretty = false;
    let mut allow_empty_output = false;
    for flag in args {
        match flag.as_str() {
            "--pretty" => pretty = true,
            "--keep-empty-output" => allow_empty_output = true,
            _ => {}
        }
    }

    // read
    let reader = BufReader::new(File::open(&in_path)?);

    let mut seen_pair: HashSet<(String, String)> = HashSet::new();
    let mut empty_counts: HashMap<&'static str, usize> = HashMap::new();
    let mut field_non_empty: HashMap<&'static str, usize> = HashMap::new();
    let mut len_stats: HashMap<&'static str, (usize, usize)> = HashMap::new(); // (sum, max)
    let mut duplicates = 0usize;
    let mut dropped_empty_output = 0usize;  // the counter!
    let mut rows: Vec<Prompt> = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let mut row: Prompt = serde_json::from_str(&line?)?;

        // ensure ids
        if row.prompt_id.is_none() {
            row.prompt_id = Some(Uuid::new_v4().to_string());
        }
        row.prompt_count = Some(idx as u64 + 1);

        // duplicate detection (case & space agnostic)
        let norm = |s: &str| s.split_whitespace().collect::<String>().to_lowercase();
        let key = (norm(&row.instruction), norm(&row.input));
        if !seen_pair.insert(key) {
            duplicates += 1;
            continue;
        }

        // optionally drop rows whose output is empty
        if row.output.trim().is_empty() && !allow_empty_output {
            dropped_empty_output += 1;
            continue;
        }

        // empty / non-empty counts & length stats
        for (name, val) in [
            ("instruction", &row.instruction),
            ("input", &row.input),
            ("output", &row.output),
        ] {
            if val.trim().is_empty() {
                *empty_counts.entry(name).or_default() += 1;
            } else {
                *field_non_empty.entry(name).or_default() += 1;
                let stats = len_stats.entry(name).or_insert((0, 0));
                let len = val.split_whitespace().count();
                stats.0 += len;              // sum
                stats.1 = stats.1.max(len);  // max
            }
        }

        rows.push(row);
    }

    // summary
    println!("=== Alpaca sanity-check ===");
    println!("Unique rows           : {}", rows.len());
    println!("Duplicates skipped    : {duplicates}");
    println!("Dropped empty output  : {dropped_empty_output}");
    println!("–– field empties –––––––––––––––––––");
    for k in ["instruction", "input", "output"] {
        println!(
            "  {:<11} {:>6} empty / {:>6} non-empty",
            k,
            empty_counts.get(k).unwrap_or(&0),
            field_non_empty.get(k).unwrap_or(&0)
        );
    }
    println!("–– token length (non-empty rows) ––");
    for k in ["instruction", "input", "output"] {
        if let Some((sum, max)) = len_stats.get(k) {
            let count = field_non_empty.get(k).copied().unwrap_or(0);
            let mean = if count > 0 { *sum as f64 / count as f64 } else { 0.0 };
            println!("  {:<11} min 1 | mean {:>5.1} | max {max}", k, mean);
        }
    }

    // write cleaned copy
    let mut out = File::create(&out_path)?;
    for row in rows {
        if pretty {
            writeln!(out, "{}", serde_json::to_string_pretty(&row)?)?;
        } else {
            writeln!(out, "{}", serde_json::to_string(&row)?)?;
        }
    }
    println!("Cleaned JSONL written → {}", out_path);
    Ok(())
}

#[allow(dead_code)]
fn out_file_name(original: &str, pretty: bool) -> String {
    let mut p = PathBuf::from(original);
    let stem = p
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("alpaca_clean");
    let suffix = if pretty { "_dedup_pretty.jsonl" } else { "_dedup.jsonl" };
    p.set_file_name(format!("{stem}{suffix}"));
    p.to_string_lossy().to_string()
}

