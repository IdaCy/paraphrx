/*
cargo run \
    --manifest-path a_data/preproc/sanity_mmlu/Cargo.toml \
    --release -- \
    --input a_data/mmlu/general_facts/world_religions.json \
    --output a_data/mmlu/global_facts_500.json \
    --count 100 \
    --start-id 401
*/
use std::{
    fs::File,
    io::{Read, Write},
    path::PathBuf,
};

use clap::Parser;
use serde::{Deserialize, Serialize};

// Select the first N items from a JSON array, renumber `prompt_count`,
// and append them to an output file (creating it if necessary)
#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    // Path to the input JSON file
    #[arg(short, long)]
    input: PathBuf,

    // Path to the output JSON file
    #[arg(short, long)]
    output: PathBuf,

    // How many elements (from the top) you want to keep
    #[arg(short, long)]
    count: usize,

    // Value to start `prompt_count` from
    #[arg(short = 's', long = "start-id")]
    start_id: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct Item {
    instruction_original: String,
    subject: String,
    choices: Vec<String>,
    answer: serde_json::Value, // could be usize or String in other files
    prompt_count: usize,
    split: String,
}

fn main() -> anyhow::Result<()> {
    // Parse CLI
    let args = Args::parse();

    // Read and deserialize input
    let mut raw = String::new();
    File::open(&args.input)?.read_to_string(&mut raw)?;
    let mut items: Vec<Item> = serde_json::from_str(&raw)?;

    // Select and renumber
    let keep = args.count.min(items.len());
    let mut selected: Vec<Item> = items.drain(0..keep).collect();
    for (i, item) in selected.iter_mut().enumerate() {
        item.prompt_count = args.start_id + i;
    }

    // Load (or initialise) the output vector
    let mut combined: Vec<Item> = if args.output.exists() {
        let mut existing_raw = String::new();
        File::open(&args.output)?.read_to_string(&mut existing_raw)?;

        if existing_raw.trim().is_empty() {
            Vec::new() // empty file → nothing to merge
        } else {
            serde_json::from_str(&existing_raw)?
        }
    } else {
        Vec::new()
    };

    // Append the new items
    combined.extend(selected);

    // Serialise and write back
    let pretty = serde_json::to_string_pretty(&combined)?;
    File::create(&args.output)?.write_all(pretty.as_bytes())?;

    println!(
        "Appended {keep} item(s) to {}, prompt_count starting at {}. \
         File now holds {} item(s).",
        args.output.display(),
        args.start_id,
        combined.len()
    );
    Ok(())
}
