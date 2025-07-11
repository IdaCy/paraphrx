/*
cargo prepare_data \
    --paraphrases-file a_data/alpaca/merge_instructs/all.json \
    --answers-file     c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/all.json \
    --scores-file      c_assess_inf/output/alpaca_answer_scores/gemma-2-2b-it.json \
    --out-file         f_finetune/data/alpaca_gemma-2-2b-it.json

gsm8k
cargo prepare_data \
    --paraphrases-file a_data/gsm8k/paraphrases_500.json \
    --answers-file     c_assess_inf/output/gsm8k/gemma-2-2b-it/answers.json \
    --scores-file      c_assess_inf/output/gsm8k_answer_scores/gemma-2-2b-it.json \
    --out-file         f_finetune/data/gsm8k_gemma-2-2b-it.json

mmlu
cargo prepare_data \
    --paraphrases-file a_data/mmlu/paraphrases_500.json \
    --answers-file     c_assess_inf/output/mmlu/gemma-2-2b-it/answers.json \
    --scores-file      c_assess_inf/output/mmlu_answer_scores/gemma-2-2b-it.json \
    --out-file         f_finetune/data/mmlu_gemma-2-2b-it.json
*/

use anyhow::{Context, Result};
use chrono::Local;
use clap::Parser;
use log::{info, warn};
use serde::Serialize;
use serde_json::Value;
use simplelog::{Config as LogConfig, LevelFilter, WriteLogger};
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::path::PathBuf;

// CLI parameters
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[arg(long)]
    paraphrases_file: PathBuf,
    #[arg(long)]
    answers_file: PathBuf,
    #[arg(long)]
    scores_file: PathBuf,
    #[arg(long = "out-file", value_name = "PATH",
           default_value = "f_finetune/data/alpaca_gemma-2-2b-it.json")]
    out_file: PathBuf,
    #[arg(long, default_value = "logs")]
    log_dir: PathBuf,
}

// One paraphrase entry in the output file
#[derive(Debug, Serialize)]
struct ParaphraseEntry {
    instruct_type: String,
    paraphrase: String,
    answer: String,
    task_score: i64,
    ranking_for_buckets: usize, // equal scores share the same rank
    bucket: u8,                 // 1-5
}

// Output object per prompt_count
#[derive(Debug, Serialize)]
struct OutputPrompt {
    prompt_count: u64,
    instruction_original: String,
    input: String,
    output: String,
    count_in_buckets: [usize; 5], // how many paraphrases ended up in each bucket
    paraphrases: Vec<ParaphraseEntry>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // logging setup
    create_dir_all(&cli.log_dir)?;
    let ts = Local::now().format("%Y%m%d_%H%M%S");
    let log_path = cli
        .log_dir
        .join(format!("prepare_finetune_data_{ts}.log"));
    WriteLogger::init(
        LevelFilter::Info,
        LogConfig::default(),
        File::create(&log_path)?,
    )?;
    info!("Starting dataset preparation");

    // load JSON
    let paraphrase_rows: Vec<Value> = read_json_array(&cli.paraphrases_file, "paraphrases")?;
    let answer_rows: Vec<Value> = read_json_array(&cli.answers_file, "answers")?;
    let score_rows: Vec<Value> = read_json_array(&cli.scores_file, "scores")?;

    let answer_map = build_index(&answer_rows)?;
    let score_map = build_index(&score_rows)?;

    let mut output: Vec<OutputPrompt> = Vec::with_capacity(paraphrase_rows.len());
    let (mut missing_a, mut missing_s) = (0usize, 0usize);

    // merge loop
    for prompt in paraphrase_rows {
        let prompt_count = prompt
            .get("prompt_count")
            .and_then(|v| v.as_u64())
            .context("Missing prompt_count")?;

        let answers_obj = match answer_map.get(&prompt_count) {
            Some(v) => v,
            None => {
                warn!("Missing answers for prompt_count {prompt_count}");
                missing_a += 1;
                continue;
            }
        };
        let scores_obj = match score_map.get(&prompt_count) {
            Some(v) => v,
            None => {
                warn!("Missing scores for prompt_count {prompt_count}");
                missing_s += 1;
                continue;
            }
        };

        // gather paraphrases
        let mut items: Vec<(String, String, String, i64)> = Vec::new();
        for (key, val) in prompt.as_object().unwrap() {
            if !(key == "instruction_original" || key.starts_with("instruct_")) {
                continue;
            }
            let paraphrase = match val.as_str() {
                Some(s) => s.to_owned(),
                None => continue,
            };
            let answer = answers_obj
                .get(key)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned();
            let score = scores_obj
                .get(key)
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            items.push((key.clone(), paraphrase, answer, score));
        }
        if items.is_empty() {
            continue;
        }

        // rank & bucket
        items.sort_by(|a, b| b.3.cmp(&a.3)); // score desc

        // group contiguous identical scores
        let mut score_groups: Vec<(i64, Vec<_>)> = Vec::new();
        for item in items {
            match score_groups.last_mut() {
                Some((s, vec)) if *s == item.3 => vec.push(item),
                _ => score_groups.push((item.3, vec![item])),
            }
        }

        let total = score_groups.iter().map(|g| g.1.len()).sum::<usize>();
        let mut bucket_target = total / 5;
        if bucket_target == 0 {
            bucket_target = 1;
        }

        let mut bucket_idx = 0usize;
        let mut filled = 0usize;
        let mut counts = [0usize; 5];
        let mut final_entries = Vec::with_capacity(total);

        for (rank_idx, (score, vec)) in score_groups.into_iter().enumerate() {
            // assign same rank & bucket to whole vec
            for (ptype, text, answer, _) in &vec {
                final_entries.push(ParaphraseEntry {
                    instruct_type: ptype.clone(),
                    paraphrase: text.clone(),
                    answer: answer.clone(),
                    task_score: score,
                    ranking_for_buckets: rank_idx + 1,
                    bucket: (bucket_idx + 1) as u8,
                });
            }
            counts[bucket_idx] += vec.len();
            filled += vec.len();

            // advance bucket if threshold reached (but never past bucket 4)
            if bucket_idx < 4 && filled >= bucket_target {
                bucket_idx += 1;
                filled = 0;
            }
        }

        // push result row
        output.push(OutputPrompt {
            prompt_count,
            instruction_original: prompt
                .get("instruction_original")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_owned(),
            input: prompt.get("input").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
            output: prompt.get("output").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
            count_in_buckets: counts,
            paraphrases: final_entries,
        });
    }

    // write JSON
    create_dir_all(cli.out_file.parent().unwrap())?;
    serde_json::to_writer_pretty(File::create(&cli.out_file)?, &output)?;
    info!("Wrote {} prompts → {:?}", output.len(), cli.out_file);

    println!("\n=== Prep summary ===");
    println!("Processed prompts  : {}", output.len());
    println!("Missing answers    : {}", missing_a);
    println!("Missing scores     : {}", missing_s);
    println!("Output JSON        : {:?}", cli.out_file);
    println!("Log file           : {:?}", log_path);

    Ok(())
}

// helpers
fn read_json_array(path: &PathBuf, label: &str) -> Result<Vec<Value>> {
    let data: Value = serde_json::from_reader(File::open(path)?)?;
    match data {
        Value::Array(arr) => Ok(arr),
        _ => anyhow::bail!("Expected array in {label} file {:?}", path),
    }
}

fn build_index<'a>(arr: &'a [Value]) -> Result<HashMap<u64, &'a Value>> {
    let mut map = HashMap::with_capacity(arr.len());
    for v in arr {
        let pc = v
            .get("prompt_count")
            .and_then(|v| v.as_u64())
            .context("prompt_count missing in scores/answers")?;
        map.insert(pc, v);
    }
    Ok(map)
}
