/*
cargo prepare_all_data \
    --paraphrases-file a_data/alpaca/paraphrases_500.json \
    --answers-file c_assess_inf/output/alpaca/gemma-2-2b-it/answers.json \
    --scores-file c_assess_inf/output/alpaca_answer_scores/gemma-2-2b-it.json \
    --paraphrase-content-scores-file a_data/alpaca/equi_scores/scores.json \
    --paraphrase-tags-file a_data/paraphrases_tagged.json \
    --out-file f_finetune/data/all_alpaca_gemma-2-2b-it.json

gsm8k
cargo prepare_all_data \
    --paraphrases-file a_data/gsm8k/paraphrases_500.json \
    --answers-file     c_assess_inf/output/gsm8k/gemma-2-2b-it/answers.json \
    --scores-file      c_assess_inf/output/gsm8k_answer_scores/gemma-2-2b-it.json \
    --paraphrase-content-scores-file a_data/gsm8k/equi_scores/scores.json \
    --paraphrase-tags-file a_data/paraphrases_tagged.json \
    --out-file         f_finetune/data/all_gsm8k_gemma-2-2b-it.json

mmlu
cargo prepare_all_data \
    --paraphrases-file a_data/mmlu/paraphrases_500.json \
    --answers-file     c_assess_inf/output/mmlu/gemma-2-2b-it/answers.json \
    --scores-file      c_assess_inf/output/mmlu_answer_scores/gemma-2-2b-it.json \
    --paraphrase-content-scores-file a_data/mmlu/equi_scores/scores.json \
    --paraphrase-tags-file a_data/paraphrases_tagged.json \
    --out-file         f_finetune/data/all_mmlu_gemma-2-2b-it.json
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
    #[arg(long)]
    paraphrase_content_scores_file: PathBuf,
    #[arg(long)]
    paraphrase_tags_file: PathBuf,
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
    answer_scores: Vec<i64>,
    paraphrase_content_score: i64,
    tags: Vec<String>,
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
    scenarios: Option<Value>,
    count_in_buckets: [usize; 5], // how many paraphrases ended up in each bucket
    paraphrases: Vec<ParaphraseEntry>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // logging setup
    create_dir_all(&cli.log_dir)?;
    let ts = Local::now().format("%Y%m%d_%H%M%S");
    let log_path = cli.log_dir.join(format!("prepare_finetune_data_{ts}.log"));
    WriteLogger::init(
        LevelFilter::Info,
        LogConfig::default(),
        File::create(&log_path)?,
    )?;
    info!("Starting dataset preparation");

    // load JSON arrays
    let paraphrase_rows: Vec<Value> = read_json_array(&cli.paraphrases_file, "paraphrases")?;
    let answer_rows: Vec<Value> = read_json_array(&cli.answers_file, "answers")?;
    let score_rows: Vec<Value> = read_json_array(&cli.scores_file, "scores")?;
    let content_score_rows: Vec<Value> = read_json_array(&cli.paraphrase_content_scores_file, "paraphrase_content_scores")?;

    // build indices
    let answer_map = build_index(&answer_rows)?;
    let score_map = build_index(&score_rows)?;
    let content_score_map = build_index(&content_score_rows)?;

    // load tags JSON
    let tags_value: Value = serde_json::from_reader(File::open(&cli.paraphrase_tags_file)?)?;
    let mut tags_map: HashMap<String, Vec<String>> = HashMap::new();
    if let Some(obj) = tags_value.as_object() {
        for (k, v) in obj.iter() {
            if let Some(arr) = v.as_array() {
                let tag_list = arr.iter().filter_map(|t| t.as_str().map(String::from)).collect();
                tags_map.insert(k.clone(), tag_list);
            }
        }
    }

    let mut output: Vec<OutputPrompt> = Vec::with_capacity(paraphrase_rows.len());
    let (mut missing_a, mut missing_s, mut missing_c) = (0usize, 0usize, 0usize);

    // merge loop
    for prompt in paraphrase_rows {
        let prompt_count = prompt
            .get("prompt_count")
            .and_then(|v| v.as_u64())
            .context("Missing prompt_count")?;

        let answers_obj = match answer_map.get(&prompt_count) {
            Some(v) => v,
            None => { warn!("Missing answers for prompt_count {prompt_count}"); missing_a += 1; continue; }
        };
        let scores_obj = match score_map.get(&prompt_count) {
            Some(v) => v,
            None => { warn!("Missing scores for prompt_count {prompt_count}"); missing_s += 1; continue; }
        };
        let content_obj = match content_score_map.get(&prompt_count) {
            Some(v) => v,
            None => { warn!("Missing paraphrase content scores for prompt_count {prompt_count}"); missing_c += 1; continue; }
        };

        // extract optional scenarios
        let scenarios = prompt.get("scenarios").cloned();

        // gather paraphrases with their data
        struct TempEntry {
            instruct_type: String,
            paraphrase: String,
            answer: String,
            task_score: i64,
            answer_scores: Vec<i64>,
            paraphrase_content_score: i64,
            tags: Vec<String>,
        }
        let mut items: Vec<TempEntry> = Vec::new();
        for (key, val) in prompt.as_object().unwrap() {
            if !(key == "instruction_original" || key.starts_with("instruct_")) {
                continue;
            }
            if key == "instruction_original" {
                continue;
            }
            let paraphrase = val.as_str().unwrap_or_default().to_owned();
            let answer = answers_obj.get(key).and_then(|v| v.as_str()).unwrap_or_default().to_owned();
            let answer_scores: Vec<i64> = scores_obj.get(key)
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_i64()).collect())
                .unwrap_or_default();
            let task_score = answer_scores.get(0).copied().unwrap_or(0);
            let paraphrase_content_score = content_obj.get("scores")
                .and_then(|sc| sc.get(key))
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let tags = tags_map.get(key).cloned().unwrap_or_default();
            items.push(TempEntry {
                instruct_type: key.clone(),
                paraphrase,
                answer,
                task_score,
                answer_scores,
                paraphrase_content_score,
                tags,
            });
        }
        if items.is_empty() { continue; }

        // sort by task_score desc
        items.sort_by(|a, b| b.task_score.cmp(&a.task_score));

        // group contiguous identical scores
        let mut score_groups: Vec<(i64, Vec<TempEntry>)> = Vec::new();
        for item in items {
            match score_groups.last_mut() {
                Some((s, vec)) if *s == item.task_score => vec.push(item),
                _ => score_groups.push((item.task_score, vec![item])),
            }
        }

        // determine bucket sizes
        let total = score_groups.iter().map(|g| g.1.len()).sum::<usize>();
        let mut bucket_target = total / 5;
        if bucket_target == 0 { bucket_target = 1; }

        let mut bucket_idx = 0usize;
        let mut filled = 0usize;
        let mut counts = [0usize; 5];
        let mut final_entries: Vec<ParaphraseEntry> = Vec::with_capacity(total);

        for (rank_idx, (_score, group)) in score_groups.into_iter().enumerate() {
            let group_len = group.len();
            for item in group {
                final_entries.push(ParaphraseEntry {
                    instruct_type: item.instruct_type,
                    paraphrase: item.paraphrase,
                    answer: item.answer,
                    task_score: item.task_score,
                    answer_scores: item.answer_scores,
                    paraphrase_content_score: item.paraphrase_content_score,
                    tags: item.tags,
                    ranking_for_buckets: rank_idx + 1,
                    bucket: (bucket_idx + 1) as u8,
                });
            }
            counts[bucket_idx] += group_len;
            filled += group_len;
            if bucket_idx < 4 && filled >= bucket_target {
                bucket_idx += 1;
                filled = 0;
            }
        }

        // push result row
        output.push(OutputPrompt {
            prompt_count,
            instruction_original: prompt.get("instruction_original").and_then(|v| v.as_str()).unwrap_or_default().to_owned(),
            input: prompt.get("input").and_then(|v| v.as_str()).unwrap_or_default().to_owned(),
            output: prompt.get("output").and_then(|v| v.as_str()).unwrap_or_default().to_owned(),
            scenarios,
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
    println!("Missing content scores: {}", missing_c);
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
        let pc = v.get("prompt_count").and_then(|v| v.as_u64()).context("prompt_count missing in index file")?;
        map.insert(pc, v);
    }
    Ok(map)
}
