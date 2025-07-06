/*

TOTAL=$(jq '.[0]|keys_unsorted|map(select(startswith("instruct_")))|length' c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/merged/extra.json)
HALF=$((TOTAL/2))

# Build an array KEEP[] holding the first HALF instruct_* keys
mapfile -t KEEP < <(
  jq -r '.[0] | keys_unsorted[]' a_data/alpaca/slice_100/extra_slice1.json |
  grep '^instruct_'
  sort
  head -n "$HALF"
)

cargo jssplit \
    -i a_data/alpaca/slice_100/extra_slice1.json \
    -a a_data/alpaca/slice_100/extra_a_slice1.json \
    -b a_data/alpaca/slice_100/extra_b_slice1.json \
    -d prompt_count -d prompt_id -d instruction_original \
    $(printf -- '-k %s ' "${KEEP[@]}")
*/

use std::{collections::HashSet, fs, path::PathBuf};
use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{Map, Value};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    // Input JSON file containing an array of objects
    #[arg(short, long)]
    input: PathBuf,

    // First output file ("A")
    #[arg(short = 'a', long)]
    output_a: PathBuf,

    // Second output file ("B")
    #[arg(short = 'b', long)]
    output_b: PathBuf,

    // Keys to appear in *both* output JSONs (may repeat)
    #[arg(short, long, value_name = "KEY")]
    duplicate: Vec<String>,

    // Keys kept in A; any other keys go to B (may repeat)
    #[arg(short, long, value_name = "KEY")]
    keep: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Read input JSON
    let raw = fs::read_to_string(&args.input)
        .with_context(|| format!("Failed to read {}", args.input.display()))?;
    let records: Vec<Value> = serde_json::from_str(&raw)
        .with_context(|| "Input must be a JSON array of objects")?;

    // Pre‑compute sets for quick lookup
    let dup_set: HashSet<&str> = args.duplicate.iter().map(String::as_str).collect();
    let keep_set: HashSet<&str> = args.keep.iter().map(String::as_str).collect();

    let mut out_a: Vec<Value> = Vec::with_capacity(records.len());
    let mut out_b: Vec<Value> = Vec::with_capacity(records.len());

    for rec in records {
        let obj = rec.as_object().cloned().context("Expected object items")?;

        let mut a_map = Map::new();
        let mut b_map = Map::new();

        for (k, v) in obj {
            let key_str = k.as_str();
            if dup_set.contains(key_str) {
                // Duplicate key into both outputs
                a_map.insert(k.clone(), v.clone());
                b_map.insert(k, v);
            } else if keep_set.contains(key_str) {
                a_map.insert(k, v);
            } else {
                b_map.insert(k, v);
            }
        }

        out_a.push(Value::Object(a_map));
        out_b.push(Value::Object(b_map));
    }

    // Write outputs
    fs::write(&args.output_a, serde_json::to_string_pretty(&out_a)?)
        .with_context(|| format!("Failed to write {}", args.output_a.display()))?;
    fs::write(&args.output_b, serde_json::to_string_pretty(&out_b)?)
        .with_context(|| format!("Failed to write {}", args.output_b.display()))?;

    println!(
        "Wrote {} records to {} and {}",
        out_a.len(),
        args.output_a.display(),
        args.output_b.display()
    );

    Ok(())
}
