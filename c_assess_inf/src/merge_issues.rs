/*
    cargo run --release -- <OUT_FILE> <IN_FILE_1> <IN_FILE_2> [...]

    cargo merge_issues \
        c_assess_inf/output/alpaca/gemma-2-9b-it/scores_issues/extra_b_issues.json \
        c_assess_inf/output/alpaca/gemma-2-9b-it/scores_issues/extra_b_slice1.issues.json \
        c_assess_inf/output/alpaca/gemma-2-9b-it/scores_issues/extra_b_slice2.issues.json \
        c_assess_inf/output/alpaca/gemma-2-9b-it/scores_issues/extra_b_slice3.issues.json \
        c_assess_inf/output/alpaca/gemma-2-9b-it/scores_issues/extra_b_slice4.issues.json \
        c_assess_inf/output/alpaca/gemma-2-9b-it/scores_issues/extra_b_slice5.issues.json

    cargo merge_issues \
        c_assess_inf/output/alpaca/gemma-2-9b-it/scores_issues/style_issues.json \
        c_assess_inf/output/alpaca/gemma-2-9b-it/scores_issues/style_slice1.issues.json \
        c_assess_inf/output/alpaca/gemma-2-9b-it/scores_issues/style_slice2.issues.json
*/

use std::{env, fs};
use serde_json::{Value, json};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Collect command‑line arguments, skipping the binary name.
    let mut args = env::args().skip(1);

    // First positional argument is the output file.
    let out_file = args.next().ok_or("Missing <OUT_FILE> argument")?;

    // Ensure at least one input file is provided.
    if args.len() == 0 {
        eprintln!("Usage: merge_json <OUT_FILE> <IN_FILE_1> <IN_FILE_2> [...]");
        std::process::exit(1);
    }

    let mut merged: Vec<Value> = Vec::new();

    for path in args {
        let data = fs::read_to_string(&path)
            .map_err(|e| format!("Cannot read '{}': {}", path, e))?;
        let json: Value = serde_json::from_str(&data)
            .map_err(|e| format!("'{}' is not valid JSON: {}", path, e))?;

        match json {
            Value::Array(items) => merged.extend(items),
            other => {
                return Err(format!("'{}' must contain a top‑level JSON array, found {}", path, other).into());
            }
        }
    }

    let serialized = serde_json::to_string_pretty(&json!(merged))?;
    fs::write(&out_file, serialized)?;

    println!("✔ Merged {} items → {}", merged.len(), out_file);
    Ok(())
}
