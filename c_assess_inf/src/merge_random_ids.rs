/*
cargo merge_random_ids \
    c_assess_inf/output/gsm8k/Qwen2.5-3B-Instruct/scores_500/voice.json \
    c_assess_inf/output/gsm8k/Qwen2.5-3B-Instruct/scores_500/voice201_results_gemini_2_5_flash_preview_05_20.json \
    c_assess_inf/output/gsm8k/Qwen2.5-3B-Instruct/scores_500/voicelast50_results_gemini_2_5_flash_preview_05_20.json
*/

use std::{env, fs};
use std::path::{Path, PathBuf};
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // gather arguments
    let files: Vec<String> = env::args().skip(1).collect();
    if files.is_empty() {
        eprintln!("Usage: merge_json <part1.json> <part2.json> [...]");
        std::process::exit(1);
    }

    // read & collect
    let mut merged: Vec<Value> = Vec::new();
    for fname in &files {
        let text = fs::read_to_string(fname)?;
        let mut part: Vec<Value> = serde_json::from_str(&text)
            .unwrap_or_else(|_| panic!("{} is not a JSON array", fname));
        merged.append(&mut part);
    }
    println!("Collected {} objects from {} file(s)", merged.len(), files.len());

    // decide output name
    let first_path = Path::new(&files[0]);
    let parent:  &Path = first_path.parent().unwrap_or_else(|| Path::new("."));
    let stem:    &str   = first_path.file_stem()
        .and_then(|s| s.to_str()).unwrap_or("merged");
    let ext:     &str   = first_path.extension()
        .and_then(|s| s.to_str()).unwrap_or("json");

    let out_name = format!("{}_merged.{}", stem.split_once('_').unwrap_or((stem, "")).0, ext);
    let out_path: PathBuf = parent.join(out_name);

    // write
    fs::write(&out_path, serde_json::to_string_pretty(&merged)?)?;
    println!("Wrote merged file → {}", out_path.display());
    Ok(())
}
