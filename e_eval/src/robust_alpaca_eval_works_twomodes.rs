/*
implements the RobustAlpacaEval benchmark to measure LLM robustness against prompt paraphrasing

operates in two modes:

1. llm-judge:
- Compares an answer from an original prompt against an answer from a paraphrased prompt using an LLM judge
- Calculates a win/loss/tie verdict
- Requires a GOOGLE_API_KEY 

2. from-scores:
- Compares pre-computed scores for original and paraphrased answers
- Determines win/loss/tie based on the difference in a specific score metric
- No API calls needed
- needs:
  --scores-original data/scores_original.json \
  --scores-paraphrased data/scores_regime1.json \
  --output results/eval_regime1_vs_original_from_scores.json

TEST:
cargo robust_alpaca_eval llm-judge \
    --prompts b_tests/robust_alpaca/prompts.json \
    --answers-original b_tests/robust_alpaca/answers_gemmaplain.json \
    --answers-paraphrased b_tests/robust_alpaca/answers_li9x_a1_notarg.json \
    --output b_tests/robust_alpaca/robust_output_23.json \
    --judging-model gemini-2.0-flash \
    --delay-ms 4000 \
    --api-call-max 6 \
    --num-judge-votes 3 \
    --include-type instruct_polite_request \
    >> logs/robalev2_$(date +%F_%T).log 2>&1 &

running:
cargo robust_alpaca_eval llm-judge \
    --prompts a_data/alpaca/50k_phrxed.json \
    --answers-original c_assess_inf/output50k/answers.json \
    --answers-paraphrased f_finetune/output_inf_ft_50k/li9x_a1_notarg_inf.json \
    --output e_eval/output_robust_alpaca_eval/li9x_a1_notarg_inf.json \
    --judging-model gemini-2.0-flash \
    --delay-ms 4000 \
    --api-call-max 200 \
    --api-key xxx \
    --num-judge-votes 3 \
    >> logs/robalev_$(date +%F_%T).log 2>&1 &

from-scores:
cargo robust_alpaca_eval from-scores \
    --prompts a_data/alpaca/50k_phrxed.json \
    --scores-original c_assess_inf/output/alpaca_answer_scores/gemma-2-2b-it.json \
    --scores-paraphrased f_finetune/output_inf_ft_50k_scores/li9x_a1_notarg_inf.json \
    --output e_eval/output_robust_alpaca_eval/li9x_a1_notarg_inf_from_scores.json \
    --log-name SCORING_FROM_SCORES \
    >> logs/robalev_from_scores_$(date +%F_%T).log 2>&1 &
*/

use anyhow::{anyhow, Context, Result};
use chrono::Local;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::time::sleep;

const ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";

// Logger
struct Logger {
    writer: BufWriter<fs::File>,
}
impl Logger {
    fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        let file = fs::OpenOptions::new().create(true).append(true).open(p)?;
        Ok(Self { writer: BufWriter::new(file) })
    }
    fn log(&mut self, msg: &str) {
        let ts = Local::now().format("%Y-%m-%d %H:%M%S");
        let _ = writeln!(self.writer, "[{ts}] {msg}");
        println!("[{ts}] {msg}");
        let _ = self.writer.flush();
    }
}

// Serde Deserialization Helpers
fn de_prompt_count<'de, D>(de: D) -> std::result::Result<u32, D::Error>
where D: serde::Deserializer<'de> {
    struct Visitor;
    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = u32;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("integer or string for prompt_count")
        }
        fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> { Ok(v as u32) }
        fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
            v.parse::<u32>().map_err(|_| E::custom(format!("invalid prompt_count {v}")))
        }
    }
    de.deserialize_any(Visitor)
}

// Data Structures
#[derive(Debug, Deserialize, Clone)]
struct PromptRecord {
    #[serde(alias = "prompt_count", deserialize_with = "de_prompt_count")]
    prompt_count: u32,
    #[serde(alias = "instruction", alias = "instruction_original")]
    instruction_original: String,
    #[serde(flatten)]
    paraphrases: JsonMap<String, Value>,
    #[serde(default)]
    input: String,
}

#[derive(Debug, Deserialize, Clone)]
struct AnswerRecord {
    #[serde(alias = "prompt_count", deserialize_with = "de_prompt_count")]
    prompt_count: u32,
    #[serde(flatten)]
    answers: JsonMap<String, Value>,
}

#[derive(Debug, Deserialize, Clone)]
struct ScoreRecord {
    #[serde(alias = "prompt_count", deserialize_with = "de_prompt_count")]
    prompt_count: u32,
    #[serde(flatten)]
    scores: JsonMap<String, Value>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
enum Verdict {
    Win,
    Loss,
    Tie,
}

// CLI Definition
#[derive(Parser, Debug)]
#[command(version, about = "RobustAlpacaEval: Evaluate LLM robustness to paraphrasing")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    LlmJudge(LlmJudgeArgs),
    FromScores(FromScoresArgs),
}

#[derive(Parser, Debug)]
struct LlmJudgeArgs {
    #[arg(long="prompts", value_name = "PROMPTS")]
    prompts: PathBuf,
    #[arg(long="answers-original", value_name = "ANSWERS_ORIGINAL")]
    answers_original: PathBuf,
    #[arg(long="answers-paraphrased", value_name = "ANSWERS_PARAPHRASED")]
    answers_paraphrased: PathBuf,
    #[arg(long="output", value_name = "OUTPUT")]
    output: PathBuf,
    #[arg(long, default_value = "gemini-1.5-flash-latest")]
    judging_model: String,
    #[arg(long, default_value = "SCORING_LLM_JUDGE")]
    log_name: String,
    #[arg(long, default_value_t = 3)]
    max_attempts: u8,
    #[arg(long = "delay-ms", default_value_t = 200)]
    delay_ms: u64,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long = "api-call-max", default_value_t = 10_000)]
    api_call_max: u32,
    #[arg(long, default_value_t = 1)]
    num_judge_votes: u32,
    #[arg(long, default_value_t = 0.5)]
    judge_temperature: f32,
    #[arg(long = "include-type", value_name = "TYPE_NAME")]
    include_types: Vec<String>,
}

#[derive(Parser, Debug)]
struct FromScoresArgs {
    #[arg(long="prompts", value_name = "PROMPTS")]
    prompts: PathBuf,
    #[arg(long="scores-original", value_name = "SCORES_ORIGINAL")]
    scores_original: PathBuf,
    #[arg(long="scores-paraphrased", value_name = "SCORES_PARAPHRASED")]
    scores_paraphrased: PathBuf,
    #[arg(long="output", value_name = "OUTPUT")]
    output: PathBuf,
    #[arg(long, default_value = "SCORING_FROM_SCORES")]
    log_name: String,
    #[arg(long, default_value_t = 0)]
    score_index: usize,
    #[arg(long, default_value_t = 0.5)]
    tie_threshold: f32,
    #[arg(long = "include-type", value_name = "TYPE_NAME")]
    include_types: Vec<String>,
}

// File reading
fn read_records<T: for<'de> Deserialize<'de>>(path: &Path, logger: &mut Logger) -> Result<Vec<T>> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("[FATAL] Could not read {}: {e}", path.display());
            logger.log(&msg);
            return Err(anyhow!(msg));
        }
    };
    match serde_json::from_str(&content) {
        Ok(r) => Ok(r),
        Err(e) => {
            let msg = format!("[FATAL] JSON parse error in {}: {e}", path.display());
            logger.log(&msg);
            return Err(anyhow!(msg));
        }
    }
}

// Main Logic
#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    fs::create_dir_all("logs")?;
    
    match cli.command {
        Commands::LlmJudge(args) => run_llm_judge(args).await,
        Commands::FromScores(args) => run_from_scores(args),
    }
}

// The main loop is now driven by the paraphrased scores file
fn run_from_scores(args: FromScoresArgs) -> Result<()> {
    let ts = Local::now().format("%Y%m%d-%H%M%S");
    let log_path = Path::new("logs").join(format!("{}_{}.log", args.log_name, ts));
    let mut logger = Logger::new(&log_path)?;
    logger.log(&format!("Run started (from-scores mode) – score_index={}, tie_threshold={}", args.score_index, args.tie_threshold));
    if !args.include_types.is_empty() {
        logger.log(&format!("Filtering to only include types: {:?}", args.include_types));
    }

    let included_types: Option<HashSet<String>> = if !args.include_types.is_empty() {
        Some(args.include_types.into_iter().collect())
    } else {
        None
    };

    logger.log(&format!("Reading prompts from: {}", args.prompts.display()));
    let _prompt_map: HashMap<u32, PromptRecord> = read_records::<PromptRecord>(&args.prompts, &mut logger)?
        .into_iter().map(|r| (r.prompt_count, r)).collect();
    logger.log(&format!("Reading original scores from: {}", args.scores_original.display()));
    let scores_orig_map: HashMap<u32, ScoreRecord> = read_records::<ScoreRecord>(&args.scores_original, &mut logger)?
        .into_iter().map(|r| (r.prompt_count, r)).collect();
    logger.log(&format!("Reading paraphrased scores from: {}", args.scores_paraphrased.display()));
    let scores_para_records: Vec<ScoreRecord> = read_records(&args.scores_paraphrased, &mut logger)?;

    let mut results: HashMap<String, JsonMap<String, Value>> = load_existing_results(&args.output, &mut logger)?;
    
    logger.log(&format!("Starting evaluation for {} paraphrased score records.", scores_para_records.len()));
    let bar = ProgressBar::new(scores_para_records.len() as u64);
    bar.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})").unwrap());

    for scores_para_rec in scores_para_records {
        bar.inc(1);
        let id = scores_para_rec.prompt_count;

        let scores_orig_rec = if let Some(rec) = scores_orig_map.get(&id) { rec } else {
            logger.log(&format!("ID {id}: Original scores not found, skipping."));
            continue;
        };

        let score_original = scores_orig_rec.scores.get("instruction_original")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.get(args.score_index))
            .and_then(|s| s.as_f64())
            .map(|s| s as f32);
        
        if score_original.is_none() {
            logger.log(&format!("ID {id}: 'instruction_original' score not found or invalid."));
            continue;
        }
        let score_original = score_original.unwrap();

        let results_entry = results.entry(id.to_string()).or_insert_with(|| {
            let mut map = JsonMap::new();
            map.insert("prompt_count".to_string(), json!(id));
            map
        });
        
        let paraphrases_to_check: Vec<_> = scores_para_rec.scores.keys()
            .filter(|k| {
                if !k.starts_with("instruct_") { return false; }
                if let Some(whitelist) = &included_types { whitelist.contains(*k) } else { true }
            })
            .cloned()
            .collect();
        
        for key in paraphrases_to_check {
            if results_entry.contains_key(&key) { continue; } 

            let score_paraphrased = scores_para_rec.scores.get(&key)
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.get(args.score_index))
                .and_then(|s| s.as_f64())
                .map(|s| s as f32);
            
            if let Some(score_para) = score_paraphrased {
                let verdict = if (score_para - score_original).abs() <= args.tie_threshold {
                    Verdict::Tie
                } else if score_para > score_original {
                    Verdict::Win
                } else {
                    Verdict::Loss
                };
                results_entry.insert(key.clone(), serde_json::to_value(verdict).unwrap());
            } else {
                logger.log(&format!("ID {id}: Score for paraphrase '{key}' not found or invalid in paraphrased scores file."));
            }
        }
    }
    bar.finish();
    
    logger.log(&format!("Finished. Writing final results to {}", args.output.display()));
    save_results(&results, &args.output)?;
    
    logger.log("\n--- Final Report (from-scores) ---");
    print_summary_report(&results, &mut logger);

    Ok(())
}


// main loop
async fn run_llm_judge(args: LlmJudgeArgs) -> Result<()> {
    let ts = Local::now().format("%Y%m%d-%H%M%S");
    let log_path = Path::new("logs").join(format!("{}_{}.log", args.log_name, ts));
    let mut logger = Logger::new(&log_path)?;
    logger.log(&format!(
        "Run started (llm-judge mode) – model={} judge_votes={}",
        args.judging_model, args.num_judge_votes
    ));
    if !args.include_types.is_empty() {
        logger.log(&format!("Filtering to only include types: {:?}", args.include_types));
    }

    let included_types: Option<HashSet<String>> = if !args.include_types.is_empty() {
        Some(args.include_types.into_iter().collect())
    } else {
        None
    };

    logger.log(&format!("Reading prompts from: {}", args.prompts.display()));
    let prompt_map: HashMap<u32, PromptRecord> = read_records::<PromptRecord>(&args.prompts, &mut logger)?
        .into_iter().map(|r| (r.prompt_count, r)).collect();
    logger.log(&format!("Reading original answers from: {}", args.answers_original.display()));
    let ans_orig_map: HashMap<u32, AnswerRecord> = read_records::<AnswerRecord>(&args.answers_original, &mut logger)?
        .into_iter().map(|r| (r.prompt_count, r)).collect();
    logger.log(&format!("Reading paraphrased answers from: {}", args.answers_paraphrased.display()));
    let ans_para_records: Vec<AnswerRecord> = read_records(&args.answers_paraphrased, &mut logger)?;

    let api_key = args.api_key.clone().or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .context("Provide --api-key or set GOOGLE_API_KEY")?;
    let client = build_client()?;

    let mut results: HashMap<String, JsonMap<String, Value>> = load_existing_results(&args.output, &mut logger)?;
    
    logger.log(&format!("Starting evaluation for {} paraphrased answer records.", ans_para_records.len()));
    let bar = ProgressBar::new(ans_para_records.len() as u64);
    bar.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})").unwrap());

    let mut api_calls_used = 0u32;
    'outer: for ans_para_rec in ans_para_records {
        bar.inc(1);
        let id = ans_para_rec.prompt_count;

        let prompt_record = if let Some(rec) = prompt_map.get(&id) { rec } else {
            logger.log(&format!("ID {id}: Corresponding prompt record not found, skipping."));
            continue;
        };
        let ans_orig_rec = if let Some(rec) = ans_orig_map.get(&id) { rec } else {
            logger.log(&format!("ID {id}: Original answer record not found, skipping."));
            continue;
        };

        let answer_original = ans_orig_rec.answers.get("instruction_original")
            .and_then(Value::as_str).unwrap_or("").trim();
        if answer_original.is_empty() {
            logger.log(&format!("ID {id}: Answer for 'instruction_original' is empty, skipping."));
            continue;
        }

        let paraphrases_to_check: Vec<_> = ans_para_rec.answers.keys()
            .filter(|k| {
                if !k.starts_with("instruct_") { return false; }
                if let Some(whitelist) = &included_types { whitelist.contains(*k) } else { true }
            })
            .cloned()
            .collect();

        for key in paraphrases_to_check {
            if api_calls_used >= args.api_call_max {
                logger.log("API call limit reached -> aborting early");
                break 'outer;
            }

            if results.get(&id.to_string()).and_then(|m| m.get(&key)).is_some() {
                continue;
            }

            let paraphrase_instruction = prompt_record.paraphrases.get(&key)
                .and_then(Value::as_str).unwrap_or("").trim();
            let paraphrase_answer = ans_para_rec.answers.get(&key)
                .and_then(Value::as_str).unwrap_or("").trim();

            if paraphrase_instruction.is_empty() {
                 logger.log(&format!("ID {id}: Instruction for paraphrase '{key}' is empty in prompts file, skipping."));
                 continue;
            }
            if paraphrase_answer.is_empty() {
                logger.log(&format!("ID {id}: Answer for paraphrase '{key}' is empty, skipping."));
                continue;
            }

            let full_instruction = if prompt_record.input.is_empty() {
                prompt_record.instruction_original.clone()
            } else {
                format!("{}\n\n[Input]\n{}", prompt_record.instruction_original, prompt_record.input)
            };
            
            let prompt_for_judge = build_judge_prompt(
                &full_instruction, 
                paraphrase_instruction, 
                answer_original, 
                paraphrase_answer
            );

            let mut votes = HashMap::new();
            for vote_num in 1..=args.num_judge_votes {
                let mut success = false;
                for attempt in 1..=args.max_attempts {
                    match query_gemini_for_judgment(&client, &api_key, &args.judging_model, &prompt_for_judge, args.judge_temperature).await {
                        Ok(verdict) => {
                            *votes.entry(verdict).or_insert(0) += 1;
                            success = true;
                            break;
                        }
                        Err(e) if attempt < args.max_attempts => {
                            logger.log(&format!("ID {id} key {key}: API vote {vote_num} attempt {attempt} failed: {e}. Retrying..."));
                            let backoff = 500u64 * 2u64.pow(attempt as u32);
                            sleep(Duration::from_millis(backoff)).await;
                        }
                        Err(e) => {
                            logger.log(&format!("ID {id} key {key}: All API attempts failed for vote {vote_num}: {e}"));
                            break;
                        }
                    }
                }
                if !success {
                     logger.log(&format!("ID {id} key {key}: Skipping due to vote failure."));
                     continue; 
                }
                api_calls_used += 1;
                if args.delay_ms > 0 && args.num_judge_votes > 1 { sleep(Duration::from_millis(args.delay_ms)).await; }
            }

            if let Some((final_verdict, _)) = votes.into_iter().max_by_key(|&(_, count)| count) {
                 logger.log(&format!("ID {id} key {key}: Final verdict is {:?}.", final_verdict));
                 let entry = results.entry(id.to_string()).or_insert_with(|| {
                    let mut map = JsonMap::new();
                    map.insert("prompt_count".to_string(), json!(id));
                    map
                 });
                 entry.insert(key.clone(), serde_json::to_value(final_verdict)?);
            }

            if let Err(e) = save_results(&results, &args.output) {
                logger.log(&format!("[ERROR] Failed to save intermediate results to {}: {}", args.output.display(), e));
            }

            if args.delay_ms > 0 { sleep(Duration::from_millis(args.delay_ms)).await; }
        }
    }
    bar.finish();

    logger.log(&format!("Finished. Writing final results to {}", args.output.display()));
    save_results(&results, &args.output)?;
    
    logger.log("\n--- Final Report (llm-judge) ---");
    print_summary_report(&results, &mut logger);

    Ok(())
}


// Utility Functions
fn load_existing_results(path: &Path, logger: &mut Logger) -> Result<HashMap<String, JsonMap<String, Value>>> {
    if path.exists() {
        logger.log(&format!("Loading existing output for resume mode: {}", path.display()));
        let content = fs::read_to_string(path)?;
        if content.trim().is_empty() {
             logger.log("Output file is empty. Starting fresh.");
             return Ok(HashMap::new());
        }
        let items: Vec<JsonMap<String, Value>> = serde_json::from_str(&content)
            .with_context(|| format!("Could not parse existing results file: {}", path.display()))?;
        items
            .into_iter()
            .map(|obj| {
                let id = obj.get("prompt_count")
                    .and_then(|v| v.as_u64().or(v.as_str().and_then(|s| s.parse().ok())))
                    .ok_or_else(|| anyhow!("Missing or invalid prompt_count in existing output"))?
                    .to_string();
                Ok((id, obj))
            })
            .collect::<Result<_, _>>()
    } else {
        logger.log("No existing output file found. Starting fresh.");
        Ok(HashMap::new())
    }
}

fn save_results(results: &HashMap<String, JsonMap<String, Value>>, output_path: &Path) -> Result<()> {
    let mut vec_out: Vec<JsonMap<String, Value>> = results.values().cloned().collect();
    vec_out.sort_by_key(|m| m.get("prompt_count").and_then(Value::as_u64).unwrap_or(0));
    let json_string = serde_json::to_string_pretty(&vec_out)?;
    if let Some(p) = output_path.parent() { fs::create_dir_all(p)?; }
    fs::write(output_path, json_string)?;
    Ok(())
}

fn print_summary_report(results: &HashMap<String, JsonMap<String, Value>>, logger: &mut Logger) {
    let mut overall_counts = BTreeMap::new();
    let mut by_type_counts: BTreeMap<String, BTreeMap<Verdict, u32>> = BTreeMap::new();

    for item in results.values() {
        for (key, value) in item {
            if key == "prompt_count" { continue; }
            if let Ok(verdict) = serde_json::from_value::<Verdict>(value.clone()) {
                *overall_counts.entry(verdict).or_insert(0) += 1;
                *by_type_counts.entry(key.clone()).or_default().entry(verdict).or_insert(0) += 1;
            }
        }
    }

    let total = overall_counts.values().sum::<u32>();
    if total == 0 {
        logger.log("No results to report.");
        return;
    }

    let wins = *overall_counts.get(&Verdict::Win).unwrap_or(&0);
    let ties = *overall_counts.get(&Verdict::Tie).unwrap_or(&0);
    let losses = *overall_counts.get(&Verdict::Loss).unwrap_or(&0);
    
    let win_rate = (wins + ties) as f32 / total as f32 * 100.0;
    
    logger.log("\n--- Overall Results ---");
    logger.log(&format!("Total Comparisons: {}", total));
    logger.log(&format!("Win Rate (Win + Tie): {:.2}%", win_rate));
    logger.log(&format!("  - Wins (Paraphrase > Original): {} ({:.2}%)", wins, wins as f32 / total as f32 * 100.0));
    logger.log(&format!("  - Ties (Paraphrase ≈ Original): {} ({:.2}%)", ties, ties as f32 / total as f32 * 100.0));
    logger.log(&format!("  - Losses (Paraphrase < Original): {} ({:.2}%)", losses, losses as f32 / total as f32 * 100.0));


    logger.log("\n--- Win Rate by Paraphrase Type ---");
    let mut sorted_types: Vec<_> = by_type_counts.into_iter().collect();
    sorted_types.sort_by_key(|(key, _)| key.clone());

    for (key, counts) in sorted_types {
        let type_total = counts.values().sum::<u32>();
        let type_wins = *counts.get(&Verdict::Win).unwrap_or(&0);
        let type_ties = *counts.get(&Verdict::Tie).unwrap_or(&0);
        let type_win_rate = (type_wins + type_ties) as f32 / type_total as f32 * 100.0;
        logger.log(&format!("- {:<30}: {:.2}%  (n={})", key, type_win_rate, type_total));
    }
}

fn build_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Ok(reqwest::Client::builder().default_headers(headers).build()?)
}

fn build_judge_prompt(original_instruction: &str, paraphrase_instruction: &str, answer_a: &str, answer_b: &str) -> String {
    format!(
r#"You are an expert evaluator. Your task is to compare two model answers based on a user's original instruction. You will be given the original instruction, the paraphrased instruction that generated Answer B, and the two answers to compare.

Your goal is to determine which answer is a better response to the **ORIGINAL instruction**. Do not penalize an answer for not following the specific style of the paraphrased prompt if it still fulfills the original instruction's core request.

[Original Instruction]
{original_instruction}

[Paraphrased Instruction that generated Answer B]
{paraphrase_instruction}

[Answer A (from Original Instruction)]
{answer_a}

---

[Answer B (from Paraphrased Instruction)]
{answer_b}

---

[Evaluation Task]
Compare Answer A and Answer B. Which answer is better at fulfilling the [Original Instruction]?

- If Answer A is significantly better, choose "A".
- If Answer B is significantly better, choose "B".
- If they are of similar quality or one is only slightly better, choose "Tie".

You must return ONLY a single, valid JSON object with a single key "verdict". The value must be one of three strings: "A", "B", or "Tie". Do not add any explanation, markdown, or surrounding text.

Example of a valid response:
{{
  "verdict": "B"
}}
"#)
}

#[derive(Deserialize)]
struct JudgeResponse {
    verdict: String,
}

async fn query_gemini_for_judgment(
    client: &reqwest::Client,
    key: &str,
    model: &str,
    prompt: &str,
    temperature: f32,
) -> Result<Verdict> {
    let url = format!("{ENDPOINT}/models/{model}:generateContent?key={key}");
    let body = json!({
        "contents": [{
            "role": "user",
            "parts": [{ "text": prompt }]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": temperature,
        }
    });

    let resp = client.post(&url).json(&body).send().await?;
    let status = resp.status();
    let resp_text = resp.text().await?;

    if !status.is_success() {
        return Err(anyhow!("API Error {}: {}", status, resp_text));
    }

    let resp_json: Value = serde_json::from_str(&resp_text)
        .with_context(|| format!("Failed to parse Gemini response shell as JSON: {resp_text}"))?;

    let json_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| anyhow!("Unexpected response structure: `text` field not found in {resp_json}"))?;

    let judge_response: JudgeResponse = serde_json::from_str(json_text.trim())
        .with_context(|| format!("Failed to parse the inner judge JSON content: {json_text}"))?;
    
    match judge_response.verdict.to_uppercase().as_str() {
        "A" => Ok(Verdict::Loss), // Original answer (A) won, so the paraphrase (B) is a Loss
        "B" => Ok(Verdict::Win),  // Paraphrased answer (B) won, so it's a Win
        "TIE" => Ok(Verdict::Tie),
        _ => Err(anyhow!("Invalid verdict received from judge: {}", judge_response.verdict)),
    }
}
