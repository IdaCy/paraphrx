/*

cargo merge_any_ids \
    -i \
<<<<<<< HEAD
    f_finetune/data/all_gsm8k_gemma-2-2b-it_all_ppl_part1.json \
    f_finetune/data/all_gsm8k_gemma-2-2b-it_all_ppl_part2.json \
    f_finetune/data/all_gsm8k_gemma-2-2b-it_all_ppl_part3.json \
    f_finetune/data/all_gsm8k_gemma-2-2b-it_all_ppl_part4.json \
    f_finetune/data/all_gsm8k_gemma-2-2b-it_all_ppl_part5.json \
    -o e_eval/data/gsm8k_gemma-2-2b-it.json
=======
<<<<<<< HEAD
    f_finetune/data/all_mmlu_gemma-2-2b-it_part1.json \
    f_finetune/data/all_mmlu_gemma-2-2b-it_part2.json \
    f_finetune/data/all_mmlu_gemma-2-2b-it_part3.json \
    f_finetune/data/all_mmlu_gemma-2-2b-it_part4.json \
    f_finetune/data/all_mmlu_gemma-2-2b-it_part5.json \
    -o f_finetune/data/all_mmlu_gemma-2-2b-it.json

=======
    f_finetune/data/all_mmlu_gemma-2-2b-it_all_ppl_part1.json \
    f_finetune/data/all_mmlu_gemma-2-2b-it_all_ppl_part2.json \
    f_finetune/data/all_mmlu_gemma-2-2b-it_all_ppl_part3.json \
    f_finetune/data/all_mmlu_gemma-2-2b-it_all_ppl_part4.json \
    f_finetune/data/all_mmlu_gemma-2-2b-it_all_ppl_part5.json \
    -o e_eval/data/mmlu_gemma-2-2b-it.json
>>>>>>> df48f8702ce53a603bdccc271ac1f51dd9d9ad96
    
>>>>>>> 53684f5f256e1e0ea6976545a1fe6f089a8dc9d7
*/

use std::{
    collections::BTreeMap,
<<<<<<< HEAD
    env,
=======
>>>>>>> 53684f5f256e1e0ea6976545a1fe6f089a8dc9d7
    fs,
    path::{Path, PathBuf},
};
use clap::{Arg, Command};
use serde_json::Value;

<<<<<<< HEAD
// helper to get a pretty name for a JSON Value variant
fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null      => "Null",
        Value::Bool(_)   => "Bool",
        Value::Number(_) => "Number",
        Value::String(_) => "String",
        Value::Array(_)  => "Array",
        Value::Object(_) => "Object",
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build the CLI
    let matches = Command::new("merge_any_ids")
=======
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build the CLI
    let matches = Command::new("merge_random_ids")
>>>>>>> 53684f5f256e1e0ea6976545a1fe6f089a8dc9d7
        .version("1.0")
        .about("Merge multiple JSON array files into one, unique by `prompt_count`")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .help("Input JSON file (array). Can be used multiple times.")
                .required(true)
<<<<<<< HEAD
                .num_args(1..),
=======
                .num_args(1..) // 1 or more
>>>>>>> 53684f5f256e1e0ea6976545a1fe6f089a8dc9d7
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output file path")
<<<<<<< HEAD
                .num_args(1),
        )
        .get_matches();

    // Collect input filenames
=======
                .num_args(1)
        )
        .get_matches();

    // collect input filenames
>>>>>>> 53684f5f256e1e0ea6976545a1fe6f089a8dc9d7
    let inputs: Vec<_> = matches
        .get_many::<String>("input")
        .unwrap()
        .map(|s| s.as_str())
        .collect();

<<<<<<< HEAD
    // Show where we're running
    let cwd = env::current_dir().expect("couldn't get current dir");
    eprintln!("🔍 Working directory: {}", cwd.display());

    // Read, validate, and collect all objects from each file
    let mut all_objs: Vec<Value> = Vec::new();
    for fname in &inputs {
        let path = Path::new(fname);
        if !path.exists() {
            panic!(
                "❌ File not found: `{}`\n   → looked in `{}`",
                fname,
                cwd.display()
            );
        }

        let text = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("Failed to read `{}`: {} (cwd: {})", fname, e, cwd.display()));

        // Detect completely empty (or whitespace‐only) files
        if text.trim().is_empty() {
            panic!(
                "❌ `{}` is empty (no content to parse) (cwd: {})",
                fname,
                cwd.display()
            );
        }

        let json: Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("Failed to parse `{}` as JSON: {} (cwd: {})", fname, e, cwd.display()));

        match json {
            Value::Array(arr) => {
                eprintln!("  • {} → {} entries", fname, arr.len());
                all_objs.extend(arr);
            }
            other => {
                panic!(
                    "`{}`: top‑level JSON is `{}`, expected `Array`",
                    fname,
                    json_type_name(&other)
                );
            }
        }
    }
    eprintln!(
        "Collected {} objects from {} file(s)",
        all_objs.len(),
        inputs.len()
    );

    // Deduplicate by `prompt_count` (first‑seen wins)
=======
    // read & merge all top‑level objects into one Vec<Value>
    let mut all_objs: Vec<Value> = Vec::new();
    for fname in &inputs {
        let text = fs::read_to_string(fname)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", fname, e));
        let part: Vec<Value> = serde_json::from_str(&text)
            .unwrap_or_else(|_| panic!("{} is not a JSON array", fname));
        all_objs.extend(part);
    }
    eprintln!("Collected {} objects from {} file(s)", all_objs.len(), inputs.len());

    // dedupe by `prompt_count`, preserving full objects
>>>>>>> 53684f5f256e1e0ea6976545a1fe6f089a8dc9d7
    let mut by_id: BTreeMap<u64, Value> = BTreeMap::new();
    for obj in all_objs {
        if let Value::Object(ref map) = obj {
            if let Some(Value::Number(n)) = map.get("prompt_count") {
                if let Some(id) = n.as_u64() {
<<<<<<< HEAD
=======
                    // first‑seen wins; later duplicates are ignored
>>>>>>> 53684f5f256e1e0ea6976545a1fe6f089a8dc9d7
                    by_id.entry(id).or_insert(obj);
                    continue;
                }
            }
        }
<<<<<<< HEAD
        // skip entries without a numeric "prompt_count"
=======
        // skip any entry without a numeric "prompt_count"
>>>>>>> 53684f5f256e1e0ea6976545a1fe6f089a8dc9d7
    }

    let unique_sorted: Vec<Value> = by_id.into_iter().map(|(_, v)| v).collect();
    eprintln!("Reduced to {} unique prompt_count IDs", unique_sorted.len());

<<<<<<< HEAD
    // Determine output path
=======
    // determine output path
>>>>>>> 53684f5f256e1e0ea6976545a1fe6f089a8dc9d7
    let out_path = if let Some(o) = matches.get_one::<String>("output") {
        PathBuf::from(o)
    } else {
        let first = Path::new(&inputs[0]);
        let parent = first.parent().unwrap_or_else(|| Path::new("."));
        let stem = first
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("merged");
        let ext = first
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("json");
        let base = stem.split_once('_').map(|(a, _)| a).unwrap_or(stem);
        parent.join(format!("{base}_merged.{ext}"))
    };

<<<<<<< HEAD
    // Write out as a pretty-printed JSON array
=======
    // write out as a pretty-printed JSON array
>>>>>>> 53684f5f256e1e0ea6976545a1fe6f089a8dc9d7
    fs::write(&out_path, serde_json::to_string_pretty(&unique_sorted)?)?;
    println!("Wrote merged file → {}", out_path.display());

    Ok(())
}
