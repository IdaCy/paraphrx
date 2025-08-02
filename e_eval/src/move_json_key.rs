/*
cargo move_json_key \
    a_data/alpaca/50k_phrxed.json \
    c_assess_inf/output50k/answers_plain.json \
    c_assess_inf/output50k/answers_w_output_proc.json
*/

use std::{collections::HashMap, env, error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    // Expect: <program> <json1> <json2> <out>
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <json1> <json2> <out>", args[0]);
        std::process::exit(1);
    }

    // Read and parse the two input files
    let json1: Vec<serde_json::Value> = serde_json::from_str(&fs::read_to_string(&args[1])?)?;
    let mut json2: Vec<serde_json::Value> = serde_json::from_str(&fs::read_to_string(&args[2])?)?;

    // Build a lookup table: prompt_count -> output, tracking JSON1 stats
    let mut output_by_count: HashMap<i64, serde_json::Value> = HashMap::new();
    let mut total_json1 = 0;
    let mut valid_json1 = 0;
    let mut invalid_json1 = 0;

    for record in json1 {
        total_json1 += 1;
        if let (Some(count), Some(output)) = (
            record.get("prompt_count").and_then(|v| v.as_i64()),
            record.get("output"),
        ) {
            output_by_count.insert(count, output.clone());
            valid_json1 += 1;
        } else {
            invalid_json1 += 1;
        }
    }

    // Merge into JSON2, tracking stats
    let total_json2 = json2.len();
    let mut inserted = 0;
    let mut missing_match = 0;
    let mut json2_missing_count = 0;

    for record in &mut json2 {
        match record.get("prompt_count").and_then(|v| v.as_i64()) {
            Some(count) => {
                if let Some(output) = output_by_count.get(&count) {
                    if let Some(obj) = record.as_object_mut() {
                        obj.insert("output".into(), output.clone());
                        inserted += 1;
                    }
                } else {
                    missing_match += 1;
                }
            }
            None => {
                json2_missing_count += 1;
            }
        }
    }

    // Write merged JSON2 back to disk
    fs::write(&args[3], serde_json::to_string_pretty(&json2)?)?;

    // Print summary
    println!(
        "JSON1 entries: total={}, valid={}, invalid={}",
        total_json1, valid_json1, invalid_json1
    );
    println!(
        "JSON2 records processed: total={}, inserted_outputs={}, missing_matches={}, missing_prompt_count={}",
        total_json2, inserted, missing_match, json2_missing_count
    );

    Ok(())
}
