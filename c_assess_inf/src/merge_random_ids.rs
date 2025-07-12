/*
cargo merge_random_ids \
    c_assess_inf/output/gsm8k/Qwen2.5-3B-Instruct/scores_500/voice.json \
    c_assess_inf/output/gsm8k/Qwen2.5-3B-Instruct/scores_500/voice201_results_gemini_2_5_flash_preview_05_20.json \
    c_assess_inf/output/gsm8k/Qwen2.5-3B-Instruct/scores_500/voicelast50_results_gemini_2_5_flash_preview_05_20.json


cargo merge_random_ids \
    -i \
    f_finetune/data/output_splits_mmlu/buckets_1-5_train_part1.json \
    f_finetune/data/output_splits_mmlu/buckets_1-5_train_part2.json \
    f_finetune/data/output_splits_mmlu/buckets_1-5_train_part3.json \
    -o f_finetune/data/output_splits_mmlu/buckets_1-5_train.json
    
*/

use std::{fs, path::{Path, PathBuf}};
use clap::{Arg, Command};
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build the CLI
    let matches = Command::new("merge_random_ids")
        .version("1.0")
        .about("Merge multiple JSON array files into one")
        .arg(Arg::new("input")
            .short('i')
            .long("input")
            .help("Input JSON file (array). Can be used multiple times.")
            .required(true)
            .num_args(1..) // 1 or more
        )
        .arg(Arg::new("output")
            .short('o')
            .long("output")
            .help("Output file path")
            .num_args(1)
        )
        .get_matches();

    // collect inputs
    let inputs: Vec<_> = matches.get_many::<String>("input")
        .unwrap()
        .map(|s| s.as_str())
        .collect();

    // read & merge
    let mut merged: Vec<Value> = Vec::new();
    for fname in &inputs {
        let text = fs::read_to_string(fname)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", fname, e));
        let mut part: Vec<Value> = serde_json::from_str(&text)
            .unwrap_or_else(|_| panic!("{} is not a JSON array", fname));
        merged.append(&mut part);
    }
    println!("Collected {} objects from {} file(s)", merged.len(), inputs.len());

    // determine output path
    let out_path = if let Some(o) = matches.get_one::<String>("output") {
        PathBuf::from(o)
    } else {
        // default: use first input's stem
        let first = Path::new(&inputs[0]);
        let parent = first.parent().unwrap_or_else(|| Path::new("."));
        let stem = first.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("merged");
        let ext = first.extension()
            .and_then(|s| s.to_str())
            .unwrap_or("json");
        let base = stem.split_once('_')
            .map(|(a, _)| a)
            .unwrap_or(stem);
        parent.join(format!("{base}_merged.{ext}"))
    };

    // write out
    fs::write(&out_path, serde_json::to_string_pretty(&merged)?)?;
    println!("Wrote merged file → {}", out_path.display());

    Ok(())
}
