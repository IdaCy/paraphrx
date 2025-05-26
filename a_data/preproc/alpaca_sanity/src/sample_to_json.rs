/*
Select rows whose prompt_count in inclusive range --from <START> --to <END>
& write them as a JSON array with keys in order:
prompt_count, instruction, input, output, prompt_id.

Example:
cargo run --manifest-path a_data/preproc/alpaca_sanity/Cargo.toml \
  --bin sample_to_json -- \
  --from 301 --to 400 \
  a_data/alpaca/alpaca_5k_proc.jsonl \
  a_data/alpaca/alpaca_100_slice4.json
*/

use anyhow::Result;
use indexmap::IndexMap;
use serde_json::Value;
use std::{
    env,
    fs::File,
    io::{BufRead, BufReader, Write},
};

fn main() -> Result<()> {
    // argument parsing
    let mut from: usize = 1;
    let mut to: usize = usize::MAX;
    let mut positional: Vec<String> = Vec::new();

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--from" => from = args.next().expect("--from <N>").parse()?,
            "--to"   => to   = args.next().expect("--to <M>").parse()?,
            _        => positional.push(arg),
        }
    }
    if positional.len() != 2 {
        eprintln!("need <INPUT.jsonl> <OUTPUT.json>");
        std::process::exit(1);
    }
    let in_path  = &positional[0];
    let out_path = &positional[1];

    // filter & collect
    let reader = BufReader::new(File::open(in_path)?);
    let mut out_vec: Vec<IndexMap<String, Value>> = Vec::new();

    for line in reader.lines() {
        let v: Value = serde_json::from_str(&line?)?;
        let pc_u64 = v
            .get("prompt_count")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| anyhow::anyhow!("missing prompt_count"))?;
        let pc = pc_u64 as usize;

        if pc >= from && pc <= to {
            let mut obj: IndexMap<String, Value> = IndexMap::new();
            obj.insert("prompt_count".to_string(), Value::from(pc_u64));
            obj.insert(
                "instruction".to_string(),
                v.get("instruction").cloned().unwrap_or(Value::Null),
            );
            obj.insert(
                "input".to_string(),
                v.get("input").cloned().unwrap_or(Value::Null),
            );
            obj.insert(
                "output".to_string(),
                v.get("output").cloned().unwrap_or(Value::Null),
            );
            obj.insert(
                "prompt_id".to_string(),
                v.get("prompt_id").cloned().unwrap_or(Value::Null),
            );
            out_vec.push(obj);
        }
        if pc > to {
            break; // ordered -> stop early
        }
    }

    // write pretty JSON array
    let mut out_file = File::create(out_path)?;
    writeln!(out_file, "{}", serde_json::to_string_pretty(&out_vec)?)?;
    Ok(())
}
