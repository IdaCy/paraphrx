/*
cargo instructmerge \
    -o c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/all.json \
    -s instruction_original \
    -s prompt_id \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/answers/boundary.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/answers/context.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/answers/extra_a.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/answers/extra_b.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/answers/language.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/answers/length.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/answers/obstruction.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/answers/special_char.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/answers/style.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/answers/syntax.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/answers/tone.json \
    c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/answers/voice.json

cargo instructmerge \
    -o c_assess_inf/output/mmlu_answer_scores/Qwen2.5-3B-Instruct.json \
    -s instruction_original \
    c_assess_inf/output/mmlu/Qwen2.5-3B-Instruct/scores_500/boundary.json \
    c_assess_inf/output/mmlu/Qwen2.5-3B-Instruct/scores_500/context.json \
    c_assess_inf/output/mmlu/Qwen2.5-3B-Instruct/scores_500/extra_a.json \
    c_assess_inf/output/mmlu/Qwen2.5-3B-Instruct/scores_500/extra_b.json \
    c_assess_inf/output/mmlu/Qwen2.5-3B-Instruct/scores_500/language.json \
    c_assess_inf/output/mmlu/Qwen2.5-3B-Instruct/scores_500/length.json \
    c_assess_inf/output/mmlu/Qwen2.5-3B-Instruct/scores_500/obstruction.json \
    c_assess_inf/output/mmlu/Qwen2.5-3B-Instruct/scores_500/special_char.json \
    c_assess_inf/output/mmlu/Qwen2.5-3B-Instruct/scores_500/style.json \
    c_assess_inf/output/mmlu/Qwen2.5-3B-Instruct/scores_500/syntax.json \
    c_assess_inf/output/mmlu/Qwen2.5-3B-Instruct/scores_500/tone.json \
    c_assess_inf/output/mmlu/Qwen2.5-3B-Instruct/scores_500/voice.json
*/

use std::{collections::{HashMap, HashSet}, fs, path::PathBuf};
use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{Map, Value};

// Merge instruction-tuning JSON files by prompt_count
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    // Output JSON file
    #[arg(short, long)]
    output: PathBuf,

    // Keys that should appear only once in the merged objects
    #[arg(short, long, required = true)]
    shared: Vec<String>,

    // Input JSON files
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Read each file into a map: prompt_count ➜ object
    let mut file_maps: Vec<HashMap<i64, Map<String, Value>>> = Vec::new();

    for path in &args.inputs {
        let data = fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let array: Vec<Value> = serde_json::from_str(&data)
            .with_context(|| format!("{} is not valid JSON array", path.display()))?;

        let mut map = HashMap::new();
        for obj in array {
            let mut obj_map = obj.as_object().cloned().context("Expected JSON objects")?;
            if !obj_map.contains_key("style") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    obj_map.insert("style".into(), Value::String(stem.to_string()));
                }
            }
            let prompt_count = obj_map.get("prompt_count")
                .and_then(Value::as_i64)
                .context("`prompt_count` must be an integer")?;

            if map.insert(prompt_count, obj_map).is_some() {
                eprintln!("Warning: duplicate prompt_count {} in {} — keeping last", prompt_count, path.display());
            }
        }
        file_maps.push(map);
    }

    // Determine the set of prompt_count values common to *all* files
    let mut common: Option<HashSet<i64>> = None;
    for fmap in &file_maps {
        let keys: HashSet<i64> = fmap.keys().copied().collect();
        common = match common {
            None => Some(keys),
            Some(acc) => Some(&acc & &keys),
        };
    }
    let common = common.unwrap_or_default();

    // Log missing prompt_counts for each file
    for (idx, fmap) in file_maps.iter().enumerate() {
        for key in fmap.keys() {
            if !common.contains(key) {
                eprintln!("File {} lacks prompt_count {} present in others; skipping.", args.inputs[idx].display(), key);
            }
        }
    }

    // Merge entries
    let mut merged: Vec<Value> = Vec::new();
    for &pc in &common {
        let mut combined = Map::new();
        combined.insert("prompt_count".into(), Value::from(pc));

        for fmap in &file_maps {
            if let Some(obj) = fmap.get(&pc) {
                for (k, v) in obj {
                    // Insert shared keys only once; ensure consistency across files
                    if args.shared.contains(k) {
                        if let Some(prev) = combined.get(k) {
                            if prev != v {
                                eprintln!("Conflict on shared key '{}' for prompt_count {}: choosing first value", k, pc);
                            }
                        } else {
                            combined.insert(k.clone(), v.clone());
                        }
                    } else {
                        // Non-shared keys: later files can overwrite earlier keys if duplicate
                        combined.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        merged.push(Value::Object(combined));
    }

    // Stable sort by prompt_count for reproducibility
    merged.sort_by_key(|v| v.get("prompt_count").and_then(Value::as_i64).unwrap_or(i64::MAX));

    // Write output
    let json_out = serde_json::to_string_pretty(&merged)?;
    fs::write(&args.output, json_out)
        .with_context(|| format!("Failed to write {}", args.output.display()))?;

    println!("Merged {} prompt groups into {}", merged.len(), args.output.display());
    Ok(())
}
