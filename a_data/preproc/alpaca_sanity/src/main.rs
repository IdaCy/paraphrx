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
    instruction: String,
    #[serde(default)]
    input: String,
    #[serde(default)]
    output: String,
}

fn main() -> Result<()> {
    let in_path = std::env::args()
        .nth(1)
        .expect("pass path to JSONL as first argument");
    let reader = BufReader::new(File::open(&in_path)?);

    let mut seen_pair: HashSet<(String, String)> = HashSet::new();
    let mut empty_counts: HashMap<&'static str, usize> = HashMap::new();
    let mut duplicates = 0usize;
    let mut rows: Vec<Prompt> = Vec::new();

    for line in reader.lines() {
        let mut row: Prompt = serde_json::from_str(&line?)?;

        // add ID if missing
        if row.prompt_id.is_none() {
            row.prompt_id = Some(Uuid::new_v4().to_string());
        }

        // detect duplicate on (instruction, input)
        let key = (row.instruction.clone(), row.input.clone());
        if !seen_pair.insert(key) {
            duplicates += 1;
            continue;
        }

        // count empties
        if row.instruction.trim().is_empty() {
            *empty_counts.entry("instruction").or_default() += 1;
        }
        if row.input.trim().is_empty() {
            *empty_counts.entry("input").or_default() += 1;
        }
        if row.output.trim().is_empty() {
            *empty_counts.entry("output").or_default() += 1;
        }

        rows.push(row);
    }

    // summary
    println!("=== Alpaca sanity-check ===");
    println!("Total unique rows  : {}", rows.len());
    println!("Duplicates skipped : {}", duplicates);
    println!("Empty-field counts :");
    for (k, v) in &empty_counts {
        println!("  {:<11} {}", k, v);
    }

    // write cleaned copy
    let out_path = out_file_name(&in_path);
    let mut out = File::create(&out_path)?;
    for row in rows {
        writeln!(out, "{}", serde_json::to_string(&row)?)?;
    }
    println!("Cleaned JSONL written → {}", out_path);
    Ok(())
}

fn out_file_name(original: &str) -> String {
    let mut p = PathBuf::from(original);
    let stem = p
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("alpaca_clean");
    p.set_file_name(format!("{stem}_dedup.jsonl"));
    p.to_string_lossy().to_string()
}
