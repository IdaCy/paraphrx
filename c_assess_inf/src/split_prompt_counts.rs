/*
cargo countsplit \
    c_assess_inf/output/gsm8k/gemma-2-2b-it/answers/extra_a.json \
    c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100/extra_a_slice1.json \
    c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100/extra_a_rest1.json \
    100

cargo countsplit \
    c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100/extra_a_rest1.json \
    c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100/extra_a_slice2.json \
    c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100/extra_a_rest2.json \
    200

cargo countsplit \
    c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100/extra_a_rest2.json \
    c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100/extra_a_slice3.json \
    c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100/extra_a_rest3.json \
    300

cargo countsplit \
    c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100/extra_a_rest3.json \
    c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100/extra_a_slice4.json \
    c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100/extra_a_slice5.json \
    400

rm c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100/extra_a_rest1.json
rm c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100/extra_a_rest2.json
rm c_assess_inf/output/gsm8k/gemma-2-2b-it/answers_slice_100/extra_a_rest3.json
*/

use std::{
    env,
    fs::File,
    io::{BufReader, BufWriter},
    process,
};

use serde_json::{self, Value};

fn main() {
    // CLI parsing
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!(
            "Usage: {} <INPUT> <OUT_LOW> <OUT_HIGH> <SPLIT_ID>",
            args[0]
        );
        process::exit(1);
    }
    let input_path = &args[1];
    let out_low_path = &args[2];
    let out_high_path = &args[3];
    let split_id: i64 = args[4].parse().expect("SPLIT_ID must be an integer");

    // Read entire input JSON (expected to be an array)
    let infile = File::open(input_path).expect("Cannot open input file");
    let reader = BufReader::new(infile);
    let mut data: Vec<Value> = serde_json::from_reader(reader)
        .expect("Input must be a JSON array of objects");

    // Partition the records
    let mut low: Vec<Value> = Vec::new();
    let mut high: Vec<Value> = Vec::new();

    for v in data.drain(..) {
        // Safely grab prompt_count; treat missing / non-int as 0
        let pc = v
            .get("prompt_count")
            .and_then(Value::as_i64)
            .unwrap_or_default();

        if pc <= split_id {
            // to pc < split_id for a strict split
            low.push(v);
        } else {
            high.push(v);
        }
    }

    // Write the two outputs
    write_json(out_low_path, &low);
    write_json(out_high_path, &high);
}

fn write_json(path: &str, payload: &[Value]) {
    let outfile = File::create(path).unwrap_or_else(|e| {
        eprintln!("Cannot create {}: {}", path, e);
        process::exit(1);
    });
    let writer = BufWriter::new(outfile);
    serde_json::to_writer_pretty(writer, payload).unwrap_or_else(|e| {
        eprintln!("Failed to write {}: {}", path, e);
        process::exit(1);
    });
}
