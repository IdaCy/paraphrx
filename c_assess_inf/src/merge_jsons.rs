/*
cargo jsmerge \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/voiceice1_infresult.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/voiceice2_infresult.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/voiceice3_infresult.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/voiceice4_infresult.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/voiceice5_infresult.json \
    -o c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/voice.json
*/

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::Value;

// Merge JSON array files in the given order into one array file.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    // Paths to the input JSON files (each must be a top-level array)
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    // Path to the output file that will be created/overwritten
    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Collect all items from each file
    let mut merged_items = Vec::<Value>::new();

    for path in &args.inputs {
        let content = fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let json: Value = serde_json::from_str(&content)
            .with_context(|| format!("parsing {}", path.display()))?;

        match json {
            Value::Array(arr) => merged_items.extend(arr),
            other => bail!(
                "File {} is not a JSON array (found {:?})",
                path.display(),
                other
            ),
        }
    }

    // Write combined array
    let pretty = serde_json::to_string_pretty(&Value::Array(merged_items))?;
    fs::write(&args.output, pretty)
        .with_context(|| format!("writing {}", args.output.display()))?;

    println!(
        "Merged {} file(s) into {}",
        args.inputs.len(),
        args.output.display()
    );
    Ok(())
}
