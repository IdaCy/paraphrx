/*

cargo merge_any_ids \
    -i \
   a_data/alpaca/alpaca_10k_part1.json \
   a_data/alpaca/alpaca_10k_part2.json \
   a_data/alpaca/alpaca_10k_part3.json \
   a_data/alpaca/alpaca_10k_part4.json \
   a_data/alpaca/alpaca_10k_part5.json \
    -o a_data/alpaca/alpaca_50k.json

cargo merge_any_ids \
    -i \
    a_data/alpaca/hpcdl/alpaca_10k_part1_phrxed.json \
    a_data/alpaca/hpcdl/alpaca_10k_part1b_phrxed.json \
    a_data/alpaca/hpcdl/alpaca_10k_part1c_phrxed.json \
    a_data/alpaca/phrxed_ese/alpaca_10k_part2a_phrxed.json \
    a_data/alpaca/phrxed_ese/alpaca_10k_part2b_phrxed.json \
    a_data/alpaca/phrxed_ese/alpaca_10k_part3a_phrxed.json \
    a_data/alpaca/phrxed_borg/alpaca_10k_part3b_phrxed.json \
    a_data/alpaca/hpcdl/alpaca_10k_part4_phrxed.json \
    a_data/alpaca/hpcdl/alpaca_10k_part4b_phrxed.json \
    a_data/alpaca/phrxed_borg/alpaca_10k_part5a_phrxed.json \
    a_data/alpaca/phrxed_borg/alpaca_10k_part5b_phrxed.json \
    a_data/alpaca/phrxed_borg/alpaca_10k_part5c_phrxed.json \
    a_data/alpaca/phrxed_ese/alpaca_10k_part5d_phrxed.json \
    a_data/alpaca/phrxed_ese/alpaca_10k_part5e_phrxed.json \
    a_data/alpaca/phrxed_ese/alpaca_10k_part5f_phrxed.json \
    -o a_data/alpaca/50k_phrxed.json
*/

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use clap::{Arg, Command};
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build the CLI
    let matches = Command::new("merge_random_ids")
        .version("1.0")
        .about("Merge multiple JSON array files into one, unique by `prompt_count`")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .help("Input JSON file (array). Can be used multiple times.")
                .required(true)
                .num_args(1..) // 1 or more
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output file path")
                .num_args(1)
        )
        .get_matches();

    // collect input filenames
    let inputs: Vec<_> = matches
        .get_many::<String>("input")
        .unwrap()
        .map(|s| s.as_str())
        .collect();

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
    let mut by_id: BTreeMap<u64, Value> = BTreeMap::new();
    for obj in all_objs {
        if let Value::Object(ref map) = obj {
            if let Some(Value::Number(n)) = map.get("prompt_count") {
                if let Some(id) = n.as_u64() {
                    // first‑seen wins; later duplicates are ignored
                    by_id.entry(id).or_insert(obj);
                    continue;
                }
            }
        }
        // skip any entry without a numeric "prompt_count"
    }

    let unique_sorted: Vec<Value> = by_id.into_iter().map(|(_, v)| v).collect();
    eprintln!("Reduced to {} unique prompt_count IDs", unique_sorted.len());

    // determine output path
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

    // write out as a pretty-printed JSON array
    fs::write(&out_path, serde_json::to_string_pretty(&unique_sorted)?)?;
    println!("Wrote merged file → {}", out_path.display());

    Ok(())
}
