/*
cargo sort_merge_ids \
    f_finetune/outputs/ft_inf_scores/buckets5.json \
    f_finetune/outputs/ft_inf_scores/buckets5_part1.json \
    f_finetune/outputs/ft_inf_scores/buckets5_part2.json
*/

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Write,
};

use anyhow::{Context, Result};
use chrono::Local;
use clap::Parser;
use serde_json::{Map, Value};

// Count how many keys an object has (helper for more-keys-wins rule)
fn key_count(v: &Value) -> usize {
    v.as_object().map(|m| m.len()).unwrap_or(0)
}

// Extract an integer ID from prompt_count
fn get_id(obj: &Map<String, Value>) -> Option<i64> {
    obj.get("prompt_count").and_then(|v| {
        if let Some(s) = v.as_str() {
            s.parse::<i64>().ok()
        } else {
            v.as_i64()
        }
    })
}

// Command-line parameters
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Merge two JSON files by prompt_count, choosing the object with more keys when duplicates occur"
)]
struct Args {
    // Output file (will be created / overwritten)
    output_file: String,
    // First input JSON file (array of objects)
    input_a: String,
    // Second input JSON file (array of objects)
    input_b: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Create log file
    fs::create_dir_all("logs").context("could not create logs/ directory")?;
    let log_name = format!(
        "logs/merge_{}.log",
        Local::now().format("%Y-%m-%d_%H-%M-%S")
    );
    let mut log = File::create(&log_name).context("could not create log file")?;
    writeln!(log, "=== json_prompt_merger started at {} ===", Local::now())?;

    // Read & parse both source files
    let read_json = |path: &str| -> Result<Vec<Value>> {
        let raw = fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        serde_json::from_str(&raw).with_context(|| format!("parsing {path} as JSON array"))
    };

    let vec_a = read_json(&args.input_a)?;
    writeln!(log, "Loaded {} objects from {}", vec_a.len(), args.input_a)?;

    let vec_b = read_json(&args.input_b)?;
    writeln!(log, "Loaded {} objects from {}", vec_b.len(), args.input_b)?;

    // Merge into a BTreeMap (keeps ascending order)
    let mut merged: BTreeMap<i64, Value> = BTreeMap::new();
    let mut duplicates_seen: BTreeSet<i64> = BTreeSet::new();

    let mut insert_object = |obj: Value, source: &str| -> Result<()> {
        if let Some(map) = obj.as_object() {
            if let Some(id) = get_id(map) {
                match merged.get(&id) {
                    None => {
                        merged.insert(id, obj);
                        writeln!(log, " -> added  id={id} from {source}")?;
                    }
                    Some(existing) => {
                        duplicates_seen.insert(id);
                        let existing_keys = key_count(existing);
                        let new_keys = key_count(&obj);
                        if new_keys > existing_keys {
                            merged.insert(id, obj);
                            writeln!(
                                log,
                                " -> duplicate id={id}: REPLACED ({}→{} keys) using {source}",
                                existing_keys, new_keys
                            )?;
                        } else {
                            writeln!(
                                log,
                                " -> duplicate id={id}: kept existing ({} keys), \
                                 discarded {}-key object from {source}",
                                existing_keys, new_keys
                            )?;
                        }
                    }
                }
            } else {
                writeln!(log, " !! skipped object with missing/invalid prompt_count in {source}")?;
            }
        } else {
            writeln!(log, " !! skipped non-object element in {source}")?;
        }
        Ok(())
    };

    for v in vec_a {
        insert_object(v, &args.input_a)?;
    }
    for v in vec_b {
        insert_object(v, &args.input_b)?;
    }

    // Write merged output
    let mut out = File::create(&args.output_file)
        .with_context(|| format!("creating {}", args.output_file))?;
    let ordered: Vec<&Value> = merged.values().collect();
    serde_json::to_writer_pretty(&mut out, &ordered)
        .with_context(|| format!("writing {}", args.output_file))?;
    writeln!(
        log,
        "\nWrote {} total objects to {}",
        ordered.len(),
        args.output_file
    )?;

    // Final summary in log
    writeln!(log, "\n=== SUMMARY ===")?;
    writeln!(
        log,
        "Duplicate IDs resolved ({} total): {:?}",
        duplicates_seen.len(),
        duplicates_seen
    )?;
    writeln!(log, "Finished at {}", Local::now())?;

    Ok(())
}
