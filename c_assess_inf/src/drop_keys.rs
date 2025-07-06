/*
cargo drop_keys \
    -i a_data/alpaca/paraphrases_500_droppedid.json \
    -o a_data/alpaca/paraphrases_500_droppedid.json \
    output prompt_id
*/

use std::fs::File;
use std::io::{self, Read};
use std::process;

use clap::Parser;
use serde_json::{Value};

// Remove one or more keys from an array-of-objects JSON
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    // Optional path to the input JSON file (read stdin when omitted)
    #[arg(short, long)]
    input: Option<String>,

    // Path to the output JSON file (overwritten if it exists)
    #[arg(short, long)]
    output: String,

    // List of keys to drop – supply one or many
    keys: Vec<String>,
}

fn main() {
    let args = Args::parse();

    // Read input
    let mut raw = String::new();
    if let Some(path) = args.input {
        File::open(&path)
            .and_then(|mut f| f.read_to_string(&mut raw))
            .unwrap_or_else(|e| exit_err(format!("Cannot read {path}: {e}")));
    } else {
        io::stdin()
            .read_to_string(&mut raw)
            .unwrap_or_else(|e| exit_err(format!("Cannot read <stdin>: {e}")));
    }

    // Parse and mutate
    let mut data: Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| exit_err(format!("Bad JSON: {e}")));

    match &mut data {
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                if let Value::Object(obj) = item {
                    for k in &args.keys {
                        obj.remove(k);
                    }
                }
            }
        }
        _ => exit_err("Top-level JSON must be an array".into()),
    }

    // Write output
    File::create(&args.output)
        .and_then(|mut f| serde_json::to_writer_pretty(&mut f, &data).map_err(|e| e.into()))
        .unwrap_or_else(|e| exit_err(format!("Cannot write {}: {e}", &args.output)));
}

fn exit_err(msg: String) -> ! {
    eprintln!("{msg}");
    process::exit(1);
}
