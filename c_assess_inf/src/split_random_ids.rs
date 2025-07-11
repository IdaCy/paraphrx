/*
cargo split_random_ids \
    f_finetune/data/mmlu_gemma-2-2b-it.json \
    100 100 100 100 100
*/

use std::{env, fs};
use std::path::Path;
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("Usage: split_json <input.json> <count1> [count2 …]");
        std::process::exit(1);
    }

    let input_path = args.remove(0);
    let counts: Vec<usize> = args.iter()
        .map(|s| s.parse::<usize>()
             .expect("Counts must be positive integers"))
        .collect();

    // Read and deserialize the JSON array
    let raw = fs::read_to_string(&input_path)?;
    let records: Vec<Value> = serde_json::from_str(&raw)
        .expect("Input must be a JSON array of objects");

    // Prepare some path helpers
    let input_path = Path::new(&input_path);
    let parent = input_path.parent().unwrap_or_else(|| Path::new("."));
    let stem   = input_path.file_stem()
        .and_then(|s| s.to_str()).unwrap_or("output");
    let ext    = input_path.extension()
        .and_then(|s| s.to_str()).unwrap_or("json");

    // Split & write
    let mut start = 0;
    for (idx, &count) in counts.iter().enumerate() {
        if start >= records.len() { break; }

        let end = (start + count).min(records.len());
        write_part(&records[start..end],
                   parent, stem, ext, idx + 1)?;
        start = end;
    }

    // Leftovers
    if start < records.len() {
        write_part(&records[start..],
                   parent, stem, ext,
                   counts.len() + 1 /* suffix */)?;
        println!("(Leftover {} object(s) → *_extra.json)", records.len() - start);
    }
    Ok(())
}

fn write_part(slice: &[Value],
              dir:   &Path,
              stem:  &str,
              ext:   &str,
              part:  usize) -> Result<(), Box<dyn std::error::Error>> {

    let extra = if part > 1 && slice.len() < 1 { "_extra" } else { "" };
    let filename = format!("{stem}_part{part}{extra}.{ext}");
    let path = dir.join(filename);

    fs::write(&path, serde_json::to_string_pretty(slice)?)?;
    println!("Wrote {} objects → {}", slice.len(), path.display());
    Ok(())
}
