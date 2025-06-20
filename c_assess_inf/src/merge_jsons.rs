/*syntax

cargo jsmerge \
    c_assess_inf/output/gsm8k/Qwen1.5-1.8B/scores_slice_100/voice_slice1.json \
    c_assess_inf/output/gsm8k/Qwen1.5-1.8B/scores_slice_100/voice_slice2.json \
    c_assess_inf/output/gsm8k/Qwen1.5-1.8B/scores_slice_100/voice_slice3.json \
    c_assess_inf/output/gsm8k/Qwen1.5-1.8B/scores_slice_100/voice_slice4.json \
    c_assess_inf/output/gsm8k/Qwen1.5-1.8B/scores_slice_100/voice_slice5.json \
    -o c_assess_inf/output/gsm8k/Qwen1.5-1.8B/scores/voice.json
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
