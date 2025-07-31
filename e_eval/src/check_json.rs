/*
cargo check_json \
    -i a_data/alpaca/50k_phrxed.json
*/

use std::{collections::BTreeSet, fs};
use clap::{Arg, Command};
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("check_missing_ids")
        .version("1.0")
        .about("Find missing prompt_count IDs in a merged JSON file and report stats")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .help("Path to the merged JSON file")
                .required(true)
        )
        .get_matches();

    // Read and parse the JSON array
    let input_path = matches.get_one::<String>("input").unwrap();
    let data = fs::read_to_string(input_path)
        .expect("Failed to read input file");
    let arr: Vec<Value> = serde_json::from_str(&data)
        .expect("Input file is not a valid JSON array");

    // Collect all prompt_count values into a sorted set
    let mut ids = BTreeSet::new();
    for obj in &arr {
        if let Some(n) = obj.get("prompt_count").and_then(Value::as_u64) {
            ids.insert(n);
        }
    }

    // none found ?
    if ids.is_empty() {
        println!("No prompt_count fields found.");
        return Ok(());
    }

    // Determine min, max, and totals
    let &min = ids.iter().next().unwrap();
    let &max = ids.iter().next_back().unwrap();
    let unique_count = ids.len();
    let range_total = (max - min + 1) as usize;
    
    // Find missing IDs in the full range
    let missing: Vec<u64> = (min..=max).filter(|n| !ids.contains(n)).collect();
    let missing_count = missing.len();
    let missing_pct = (missing_count as f64) * 100.0 / (range_total as f64);

    // Report summary stats
    println!("Prompt_count ID stats for file: {}", input_path);
    println!("  Minimum ID: {}", min);
    println!("  Maximum ID: {}", max);
    println!("  Unique IDs found: {}", unique_count);
    println!("  Total IDs expected ({} to {} inclusive): {}", min, max, range_total);

    // Report missing
    if missing_count == 0 {
        println!("No missing IDs between {} and {}.", min, max);
    } else {
        println!("Missing IDs ({} total, {:.2}%):", missing_count, missing_pct);
        println!("  {}", missing.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "));    
    }

    Ok(())
}
