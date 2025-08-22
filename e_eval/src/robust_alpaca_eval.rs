/*
cargo robust_alpaca_eval \
  llm-judge \
  --prompts a_data/alpaca/50k_phrxed.json \
  --answers-reference c_assess_inf/output50k/gpt4_answers_1440.json \
  --answers-target e_eval/output_robust_alpaca_eval/answers_phrx.json \
  --output-dir e_eval/output_robust_alpaca_eval/scores \
  --api-key "xxx" \
  --api-key "xxx" \
  --max-per-key 200 \
  --judging-model gemini-2.0-flash \
  --delay-ms 4000 \
  --length-control \
  --seed 42 \
  >> e_eval/output_robust_alpaca_eval/phrx_$(date +%F_%T).log 2>&1 &
cargo robust_alpaca_eval \
  llm-judge \
  --prompts a_data/alpaca/50k_phrxed.json \
  --answers-reference c_assess_inf/output50k/gpt4_answers_1440.json \
  --answers-target e_eval/output_robust_alpaca_eval/answers_phrx.json \
  --output-dir e_eval/output_robust_alpaca_eval/scores \
  --api-key "xxx" \
  --max-per-key 200 \
  --judging-model gemini-2.0-flash \
  --delay-ms 4000 \
  --length-control \
  --seed 42 \
  >> e_eval/output_robust_alpaca_eval/phrx_$(date +%F_%T).log 2>&1 &
cargo run --release -- \
  llm-judge \
  --prompts a_data/alpaca/50k_phrxed.json \
  --answers-reference c_assess_inf/output50k/gpt4_answers_1440.json \
  --answers-target e_eval/output_robust_alpaca_eval/answers_phrx.json \
  --outout-file e_eval/output_robust_alpaca_eval/scores_phrx.json \
  --log-name phrx_eval \
  --api-key "xxx" \
  --api-key "xxx" \
  --api-key "xxx" \
  --api-key "xxx" \
  --max-per-key 200 \
  --judging-model gemini-2.0-flash \
  --delay-ms 4000 \
  --length-control \
  --seed 42 \
  > e_eval/output_robust_alpaca_eval/phrx_$(date +%F_%T).stdout.log 2>&1 &

cargo robust_alpaca_eval \
  llm-judge \
  --prompts a_data/alpaca/50k_phrxed.json \
  --answers-reference c_assess_inf/output50k/gpt4_answers_1440.json \
  --answers-target e_eval/output_robust_alpaca_eval/answers_onlylap.json \
  --output-dir e_eval/output_robust_alpaca_eval/scores_onlylap \
  --api-key "xxx" \
  --api-key "xxx" \
  --api-key "xxx" \
  --api-key "xxx" \
  --max-per-key 200 \
  --judging-model gemini-2.0-flash \
  --delay-ms 4000 \
  --length-control \
  --seed 42 \
  >> e_eval/output_robust_alpaca_eval/onlylap_$(date +%F_%T).log 2>&1 &
cargo robust_alpaca_eval \
  llm-judge \
  --prompts a_data/alpaca/50k_phrxed.json \
  --answers-reference c_assess_inf/output50k/gpt4_answers_1440.json \
  --answers-target e_eval/output_robust_alpaca_eval/answers_onlylap.json \
  --output-dir e_eval/output_robust_alpaca_eval/scores_onlylap \
  --api-key "xxx" \
  --api-key "xxx" \
  --api-key "xxx" \
  --max-per-key 200 \
  --judging-model gemini-2.0-flash \
  --delay-ms 4000 \
  --length-control \
  --seed 42 \
  >> e_eval/output_robust_alpaca_eval/onlylap_$(date +%F_%T).log 2>&1 &

cargo robust_alpaca_eval \
  llm-judge \
  --prompts a_data/alpaca/50k_phrxed.json \
  --answers-reference c_assess_inf/output50k/gpt4_answers_1440.json \
  --answers-target f_finetune/outputs_great_nolap/h_ultrafeedback_binarized_nolap/answers.json \
  --output-dir e_eval/output_robust_alpaca_eval/scores \
  --api-key "xxx" \
  --api-key "xxx" \
  --api-key "xxx" \
  --api-key "xxx" \
  --max-per-key 200 \
  --judging-model gemini-2.0-flash \
  --delay-ms 4000 \
  --length-control \
  --seed 42 \
  >> e_eval/output_robust_alpaca_eval/ultrafeedback_binarized_nolap_$(date +%F_%T).log 2>&1 &

cargo robust_alpaca_eval \
  llm-judge \
  --prompts a_data/alpaca/50k_phrxed.json \
  --answers-reference c_assess_inf/output50k/gpt4_answers_1440.json \
  --answers-target f_finetune/model/answers.json \
  --output-dir e_eval/output_robust_alpaca_eval/scores_base \
  --api-key "xxx" \
  --api-key "xxx" \
  --api-key "xxx" \
  --api-key "xxx" \
  --max-per-key 200 \
  --judging-model gemini-2.0-flash \
  --delay-ms 4000 \
  --length-control \
  --seed 42 \
  >> e_eval/output_robust_alpaca_eval/base_$(date +%F_%T).log 2>&1 &
cargo robust_alpaca_eval \
  llm-judge \
  --prompts a_data/alpaca/50k_phrxed.json \
  --answers-reference c_assess_inf/output50k/gpt4_answers_1440.json \
  --answers-target f_finetune/model/answers.json \
  --output-dir e_eval/output_robust_alpaca_eval/scores_base \
  --api-key "xxx" \
  --max-per-key 1 \
  --judging-model gemini-2.0-flash \
  --delay-ms 4000 \
  --length-control \
  --seed 42 \
  >> e_eval/output_robust_alpaca_eval/base_$(date +%F_%T).log 2>&1 &

cargo robust_alpaca_eval \
  llm-judge \
  --prompts a_data/alpaca/50k_phrxed.json \
  --answers-reference c_assess_inf/output50k/gpt4_answers_1440.json \
  --answers-target f_finetune/outputs_lap_pr/real9x_output_stable5/answers.json \
  --output-dir e_eval/output_robust_alpaca_eval/scores \
  --api-key "xxx" \
  --api-key "xxx" \
  --api-key "xxx" \
  --api-key "xxx" \
  --max-per-key 200 \
  --judging-model gemini-2.0-flash \
  --delay-ms 4000 \
  --length-control \
  --seed 42 \
  >> e_eval/output_robust_alpaca_eval/real9x_output_stable5_$(date +%F_%T).log 2>&1 &

cargo robust_alpaca_eval \
  llm-judge \
  --prompts a_data/alpaca/50k_phrxed.json \
  --answers-reference c_assess_inf/output50k/gpt4_answers_1440.json \
  --answers-target f_finetune/outputs_great_nolap/ft_spec6layer/answers.json \
  --output-dir e_eval/output_robust_alpaca_eval/scores_ft_spec6layer \
  --api-key "xxx" \
  --api-key "xxx" \
  --api-key "xxx" \
  --api-key "xxx" \
  --max-per-key 200 \
  --judging-model gemini-2.0-flash \
  --delay-ms 4000 \
  --length-control \
  --seed 42 \
  >> e_eval/output_robust_alpaca_eval/ft_spec6layer_$(date +%F_%T).log 2>&1 &
*/
use anyhow::{anyhow, Context, Result};
use chrono::Local;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
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

// Serde helpers
fn de_prompt_count<'de, D>(de: D) -> std::result::Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Visitor;
    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = u32;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("integer or string for prompt_count")
        }
        fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
            Ok(v as u32)
        }
        fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
            v.parse::<u32>()
                .map_err(|_| E::custom(format!("invalid prompt_count {v}")))
        }
    }
    de.deserialize_any(Visitor)
}

// Data structures (prompts / answers / scores / verdicts)
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

#[derive(Debug, Deserialize, Clone)]
struct VerdictRecord {
    #[serde(alias = "prompt_count", deserialize_with = "de_prompt_count")]
    prompt_count: u32,
    #[serde(flatten)]
    verdicts: JsonMap<String, Value>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
enum Verdict {
    Win,
    Loss,
    Tie,
}
impl Verdict {
    fn to_score(self) -> f64 {
        match self {
            Verdict::Win => 1.0,
            Verdict::Tie => 0.5,
            Verdict::Loss => 0.0,
        }
    }
}

// File read helper
fn read_records<T: for<'de> Deserialize<'de>>(path: &Path, logger: &mut Logger) -> Result<Vec<T>> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("[FATAL] Could not read {}: {e}", path.display());
            logger.log(&msg);
            return Err(anyhow!(msg));
        }
    };
    if content.trim().is_empty() {
        return Ok(Vec::new());
    }
    match serde_json::from_str(&content) {
        Ok(r) => Ok(r),
        Err(e) => {
            let msg = format!("[FATAL] JSON parse error in {}: {e}", path.display());
            logger.log(&msg);
            Err(anyhow!(msg))
        }
    }
}

// CLI
#[derive(Parser, Debug)]
#[command(version, about = "RobustAlpacaEval: Evaluate LLM robustness to paraphrasing")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Evaluate using an LLM as judge (RobustAlpacaEval-compatible)
    LlmJudge(LlmJudgeArgs),
    /// Evaluate from pre-computed numerical scores
    FromScores(FromScoresArgs),
    /// Report from a pre-computed Win/Loss/Tie file
    FromLlmScores(FromLlmScoresArgs),
}

#[derive(Parser, Debug)]
struct LlmJudgeArgs {
    #[arg(long = "prompts", value_name = "PROMPTS")]
    prompts: PathBuf,

    #[arg(long = "answers-reference", value_name = "ANSWERS_REFERENCE")]
    answers_reference: PathBuf,

    #[arg(long = "answers-target", value_name = "ANSWERS_TARGET", num_args = 1..)]
    answers_targets: Vec<PathBuf>,

    #[arg(long = "output-dir", value_name = "OUTPUT_DIR")]
    output_dir: PathBuf,

    #[arg(long, default_value = "gemini-1.5-flash-latest")]
    judging_model: String,

    #[arg(long, default_value_t = 1)]
    num_judge_votes: u32,

    #[arg(long, default_value_t = 0.0)]
    judge_temperature: f32,

    #[arg(long = "include-type", value_name = "TYPE_NAME")]
    include_types: Vec<String>,

    #[arg(long = "api-key", value_name = "API_KEY", num_args = 0.., value_delimiter = ',')]
    api_keys: Vec<String>,

    #[arg(long = "max-per-key", default_value_t = 10_000)]
    max_per_key: u32,

    #[arg(long = "api-call-max", default_value_t = 1_000_000)]
    api_call_max: u32,

    #[arg(long = "delay-ms", default_value_t = 100)]
    delay_ms: u64,

    /// Enable length-controlled debiasing via OLS (legacy)
    #[arg(long = "length-control")]
    length_control: bool,

    /// Limit the TOTAL number of prompt_count records across all targets
    #[arg(long = "max-records", value_name = "N")]
    max_records: Option<usize>,

    /// Deterministic shuffling for reproducibility
    #[arg(long = "seed")]
    seed: Option<u64>,
}

#[derive(Parser, Debug)]
struct FromScoresArgs {
    #[arg(long = "prompts", value_name = "PROMPTS")]
    prompts: PathBuf,
    #[arg(long = "scores-original", value_name = "SCORES_ORIGINAL")]
    scores_original: PathBuf,
    #[arg(long = "scores-paraphrased", value_name = "SCORES_PARAPHRASED")]
    scores_paraphrased: PathBuf,
    #[arg(long = "output", value_name = "OUTPUT")]
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

#[derive(Parser, Debug)]
struct FromLlmScoresArgs {
    #[arg(long = "verdicts", value_name = "VERDICTS_FILE")]
    verdicts: PathBuf,
    #[arg(long, default_value = "REPORTING_FROM_LLM_SCORES")]
    log_name: String,
    #[arg(long = "include-type", value_name = "TYPE_NAME")]
    include_types: Vec<String>,
}

// Main
#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    fs::create_dir_all("logs")?;
    match cli.command {
        Commands::LlmJudge(args) => run_llm_judge(args).await,
        Commands::FromScores(args) => run_from_scores(args),
        Commands::FromLlmScores(args) => run_from_llm_scores(args),
    }
}

// From pre-computed verdicts (unchanged from your intent; cleaned up)
fn run_from_llm_scores(args: FromLlmScoresArgs) -> Result<()> {
    let ts = Local::now().format("%Y%m%d-%H%M%S");
    let log_path = Path::new("logs").join(format!("{}_{}.log", args.log_name, ts));
    let mut logger = Logger::new(&log_path)?;
    logger.log("Run started (from-llm-scores mode)");
    let included_types: Option<HashSet<String>> =
        if !args.include_types.is_empty() { Some(args.include_types.into_iter().collect()) } else { None };

    logger.log(&format!(
        "Reading pre-computed verdicts from: {}",
        args.verdicts.display()
    ));
    let verdict_records: Vec<VerdictRecord> = read_records(&args.verdicts, &mut logger)?;

    let mut results: HashMap<String, JsonMap<String, Value>> = HashMap::new();
    for record in verdict_records {
        let id_str = record.prompt_count.to_string();
        let mut entry_map = JsonMap::new();
        entry_map.insert("prompt_count".to_string(), json!(record.prompt_count));
        for (key, value) in record.verdicts {
            if let Some(whitelist) = &included_types {
                if whitelist.contains(&key) {
                    entry_map.insert(key, value);
                }
            } else {
                entry_map.insert(key, value);
            }
        }
        results.insert(id_str, entry_map);
    }

    logger.log("\n--- Final Report (from pre-computed verdicts) ---");
    print_summary_report(&results, &mut logger);
    logger.log("Reporting complete.");
    Ok(())
}

// From pre-computed numeric scores (unchanged logic; removed stray cap code)
fn run_from_scores(args: FromScoresArgs) -> Result<()> {
    let ts = Local::now().format("%Y%m%d-%H%M%S");
    let log_path = Path::new("logs").join(format!("{}_{}.log", args.log_name, ts));
    let mut logger = Logger::new(&log_path)?;
    logger.log(&format!(
        "Run started (from-scores mode) – score_index={}, tie_threshold={}",
        args.score_index, args.tie_threshold
    ));

    let included_types: Option<HashSet<String>> =
        if !args.include_types.is_empty() { Some(args.include_types.into_iter().collect()) } else { None };

    logger.log(&format!("Reading prompts from: {}", args.prompts.display()));
    let _prompt_map: HashMap<u32, PromptRecord> = read_records::<PromptRecord>(&args.prompts, &mut logger)?
        .into_iter()
        .map(|r| (r.prompt_count, r))
        .collect();
    logger.log(&format!(
        "Reading original scores from: {}",
        args.scores_original.display()
    ));
    let scores_orig_map: HashMap<u32, ScoreRecord> =
        read_records::<ScoreRecord>(&args.scores_original, &mut logger)?
            .into_iter()
            .map(|r| (r.prompt_count, r))
            .collect();
    logger.log(&format!(
        "Reading paraphrased scores from: {}",
        args.scores_paraphrased.display()
    ));
    let scores_para_records: Vec<ScoreRecord> =
        read_records(&args.scores_paraphrased, &mut logger)?;

    let mut results: HashMap<String, JsonMap<String, Value>> =
        load_existing_results(&PathBuf::from("from_scores.json"), &mut logger)?; // ephemeral

    logger.log(&format!(
        "Starting evaluation for {} paraphrased score records.",
        scores_para_records.len()
    ));
    let bar = ProgressBar::new(scores_para_records.len() as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap(),
    );

    for scores_para_rec in scores_para_records {
        bar.inc(1);
        let id = scores_para_rec.prompt_count;
        let scores_orig_rec = if let Some(rec) = scores_orig_map.get(&id) {
            rec
        } else {
            logger.log(&format!("ID {id}: Original scores not found, skipping."));
            continue;
        };

        let score_original = scores_orig_rec
            .scores
            .get("instruction_original")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.get(args.score_index))
            .and_then(|s| s.as_f64())
            .map(|s| s as f32);
        if score_original.is_none() {
            logger.log(&format!(
                "ID {id}: 'instruction_original' score not found or invalid."
            ));
            continue;
        }
        let score_original = score_original.unwrap();

        let results_entry = results.entry(id.to_string()).or_insert_with(|| {
            let mut map = JsonMap::new();
            map.insert("prompt_count".to_string(), json!(id));
            map
        });

        let paraphrases_to_check: Vec<_> = scores_para_rec
            .scores
            .keys()
            .filter(|k| {
                if !k.starts_with("instruct_") {
                    return false;
                }
                if let Some(whitelist) = &included_types {
                    whitelist.contains(*k)
                } else {
                    true
                }
            })
            .cloned()
            .collect();
        for key in paraphrases_to_check {
            if results_entry.contains_key(&key) {
                continue;
            }
            let score_paraphrased = scores_para_rec
                .scores
                .get(&key)
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

    logger.log("\n--- Final Report (from-scores) ---");
    print_summary_report(&results, &mut logger);
    Ok(())
}

// LLM-Judge (RobustAlpacaEval-compatible)
async fn run_llm_judge(args: LlmJudgeArgs) -> Result<()> {
    // Setup logging
    let ts = Local::now().format("%Y%m%d-%H%M%S");
    fs::create_dir_all(&args.output_dir)?;
    let log_path = Path::new("logs").join(format!("SCORING_LLM_JUDGE_{}.log", ts));
    let mut logger = Logger::new(&log_path)?;

    // Global record limit across all targets (None => unlimited)
    let mut remaining_records: usize = args.max_records.unwrap_or(usize::MAX);

    // Include-filter
    let included_types: Option<HashSet<String>> =
        if !args.include_types.is_empty() { Some(args.include_types.iter().cloned().collect()) } else { None };

    // Read inputs
    logger.log(&format!("Reading prompts from: {}", args.prompts.display()));
    let prompt_map: HashMap<u32, PromptRecord> =
        read_records::<PromptRecord>(&args.prompts, &mut logger)?
            .into_iter()
            .map(|r| (r.prompt_count, r))
            .collect();

    logger.log(&format!(
        "Reading reference answers from: {}",
        args.answers_reference.display()
    ));
    let ans_ref_map: HashMap<u32, AnswerRecord> =
        read_records::<AnswerRecord>(&args.answers_reference, &mut logger)?
            .into_iter()
            .map(|r| (r.prompt_count, r))
            .collect();

    if args.answers_targets.is_empty() {
        return Err(anyhow!("Provide at least one --answers-target file"));
    }

    // API keys rotation
    let mut key_pool = ApiKeyPool::new(args.api_keys.clone(), args.max_per_key)?;
    let client = build_client()?;

    // RNG for deterministic shuffling if seed supplied
    let mut rng: StdRng = match args.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // Per-target evaluation loop
    for target_path in &args.answers_targets {
        let target_name = target_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("target")
            .to_string();
        logger.log(&format!("\n=== Evaluating target: {} ===", target_name));
        let out_path = args
            .output_dir
            .join(format!("robust_alpacaeval__{}.json", sanitize_for_fs(&target_name)));

        let all_ans_target_records: Vec<AnswerRecord> =
            read_records(target_path, &mut logger)?;

        // Apply global cap to this target's records
        let mut ans_target_records = all_ans_target_records;
        if remaining_records != usize::MAX {
            let to_take = remaining_records.min(ans_target_records.len());
            ans_target_records.truncate(to_take);
            logger.log(&format!(
                "Limiting to {} records for this target ({} remaining overall before this target).",
                to_take, remaining_records
            ));
        }
        let selected_len = ans_target_records.len();
        if selected_len == 0 {
            logger.log("No records selected for this target under the global cap; skipping target.");
            continue;
        }

        // Load/merge resume
        let mut results: HashMap<String, JsonMap<String, Value>> =
            load_existing_results(&out_path, &mut logger)?;

        // For debiasing / stats
        let mut lc_pairs_ols: Vec<(f64, f64)> = Vec::new(); // (y, len_diff)
        let mut glm_pairs: Vec<(u8, f64)> = Vec::new(); // (y_bin, len_diff) exclude ties
        let mut weighted_wr_terms: Vec<f64> = Vec::new(); // mapped confidence contributions

        // Progress bar
        logger.log(&format!(
            "Starting evaluation for {} target records.",
            ans_target_records.len()
        ));
        let bar = ProgressBar::new(ans_target_records.len() as u64);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap(),
        );

        let mut api_calls_used: u32 = 0;
        'outer: for ans_t in ans_target_records {
            bar.inc(1);
            let id = ans_t.prompt_count;

            let prompt_rec = match prompt_map.get(&id) {
                Some(p) => p,
                None => {
                    logger.log(&format!("ID {id}: prompt missing -> skip"));
                    continue;
                }
            };
            let ans_ref_rec = match ans_ref_map.get(&id) {
                Some(r) => r,
                None => {
                    logger.log(&format!("ID {id}: reference answers missing -> skip"));
                    continue;
                }
            };

            // Diagnostics about available paraphrases per item
            let prompt_instruct: HashSet<String> = prompt_rec
                .paraphrases
                .keys()
                .filter(|k| k.starts_with("instruct_"))
                .cloned()
                .collect();
            let keys_t: HashSet<String> = ans_t.answers.keys().cloned().collect();
            let keys_r: HashSet<String> = ans_ref_rec.answers.keys().cloned().collect();
            let common: HashSet<String> = keys_t
                .intersection(&keys_r)
                .cloned()
                .filter(|k| k == "instruction_original" || k.starts_with("instruct_"))
                .collect();

            if common.len() == 1 && common.contains("instruction_original") {
                logger.log(&format!(
                    "ID {id}: only instruction_original present in BOTH answers; paraphrases missing on at least one side."
                ));
            }
            // Log unexpected absences
            for k in prompt_instruct.difference(&keys_t).cloned().collect::<Vec<_>>() {
                logger.log(&format!(
                    "ID {id}: paraphrase '{k}' present in PROMPTS but missing in TARGET answers."
                ));
            }
            for k in prompt_instruct.difference(&keys_r).cloned().collect::<Vec<_>>() {
                logger.log(&format!(
                    "ID {id}: paraphrase '{k}' present in PROMPTS but missing in REFERENCE answers."
                ));
            }

            // Build list of keys to evaluate (intersection)
            let mut keys: Vec<String> = common.into_iter().collect();
            if let Some(whitelist) = &included_types {
                keys.retain(|k| {
                    if k == "instruction_original" {
                        true
                    } else if k.starts_with("instruct_") {
                        whitelist.contains(k)
                    } else {
                        false
                    }
                });
            }
            keys.sort();

            for key in keys {
                if api_calls_used >= args.api_call_max {
                    logger.log("API call limit reached -> aborting early");
                    break 'outer;
                }
                // Stop immediately if all keys are exhausted
                if key_pool.exhausted() {
                    logger.log("All API keys exhausted -> aborting early");
                    break 'outer;
                }

                // Skip if already judged
                if results
                    .get(&id.to_string())
                    .and_then(|m| m.get(&key))
                    .is_some()
                {
                    continue;
                }

                // Instruction shown to judge: the EXACT instruction for this key
                let instr = if key == "instruction_original" {
                    prompt_rec.instruction_original.clone()
                } else {
                    prompt_rec
                        .paraphrases
                        .get(&key)
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string()
                };
                if instr.trim().is_empty() {
                    logger.log(&format!(
                        "ID {id} key {key}: instruction text missing -> skip"
                    ));
                    continue;
                }

                // Optional input
                let full_instruction = if prompt_rec.input.trim().is_empty() {
                    instr
                } else {
                    format!("{instr}\n\n[Input]\n{}", prompt_rec.input)
                };

                // Answers
                let a_target = ans_t
                    .answers
                    .get(&key)
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let a_ref = ans_ref_rec
                    .answers
                    .get(&key)
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if a_target.is_empty() || a_ref.is_empty() {
                    logger.log(&format!("ID {id} key {key}: empty outputs -> skip"));
                    continue;
                }

                // Randomize order (deterministic if --seed)
                let mut pair = vec![("target", a_target.as_str()), ("reference", a_ref.as_str())];
                pair.shuffle(&mut rng);
                let ans1 = pair[0].1;
                let ans2 = pair[1].1;
                let ans1_is_target = pair[0].0 == "target";

                let prompt_for_judge = build_judge_prompt(&full_instruction, ans1, ans2);

                // Voting loop (rotates/consumes keys PER ATTEMPT)
                let mut votes: HashMap<Verdict12, u32> = HashMap::new();
                let mut vote_confidences: Vec<f64> = Vec::new(); // for weighted WR

                for vote_num in 1..=args.num_judge_votes {
                    if api_calls_used >= args.api_call_max {
                        logger.log("API call limit reached -> aborting early");
                        break 'outer;
                    }

                    let mut success = false;
                    let mut last_err: Option<anyhow::Error> = None;

                    for attempt in 1..=3u8 {
                        let api_key = match key_pool.take_key_for_call() {
                            Ok(k) => k,
                            Err(_) => {
                                logger.log("All API keys exhausted -> aborting early");
                                break 'outer;
                            }
                        };

                        match query_gemini_for_judgment(
                            &client,
                            &api_key,
                            &args.judging_model,
                            &prompt_for_judge,
                            args.judge_temperature,
                        ).await {
                            Ok((local_verdict, conf12)) => {
                                *votes.entry(local_verdict).or_insert(0) += 1;
                                vote_confidences.push(conf12.clamp(0.0, 1.0));
                                success = true;
                                break;
                            }
                            Err(e) if attempt < 3 => {
                                last_err = Some(e);
                                logger.log(&format!(
                                    "ID {id} key {key}: vote {vote_num} attempt {attempt} failed; retrying..."
                                ));
                                sleep(Duration::from_millis(300 * (attempt as u64))).await;
                            }
                            Err(e) => {
                                last_err = Some(e);
                                logger.log(&format!(
                                    "ID {id} key {key}: vote {vote_num} failed after retries."
                                ));
                                break;
                            }
                        }
                    }

                    if !success {
                        if let Some(e) = last_err {
                            logger.log(&format!("Final error: {e}"));
                        }
                        logger.log(&format!("ID {id} key {key}: skipping this vote due to failure"));
                    }

                    api_calls_used += 1;
                    if args.delay_ms > 0 && args.num_judge_votes > 1 {
                        sleep(Duration::from_millis(args.delay_ms)).await;
                    }
                }

                // Aggregate votes
                if votes.is_empty() {
                    continue;
                }
                let (verdict_12, _) = votes.into_iter().max_by_key(|&(_, c)| c).unwrap();
                // map to target/ref
                let verdict = match verdict_12 {
                    Verdict12::One => {
                        if ans1_is_target {
                            Verdict::Win
                        } else {
                            Verdict::Loss
                        }
                    }
                    Verdict12::Two => {
                        if ans1_is_target {
                            Verdict::Loss
                        } else {
                            Verdict::Win
                        }
                    }
                    Verdict12::Tie => Verdict::Tie,
                };

                // Weighted WR term (use mean confidence from votes)
                let conf12 = if !vote_confidences.is_empty() {
                    vote_confidences.iter().copied().sum::<f64>() / (vote_confidences.len() as f64)
                } else {
                    0.5
                };
                // Convert confidence on "chosen between answer1/answer2" to "confidence that TARGET > REF"
                let conf_target_better = match verdict_12 {
                    Verdict12::One => {
                        if ans1_is_target {
                            conf12
                        } else {
                            1.0 - conf12
                        }
                    }
                    Verdict12::Two => {
                        if ans1_is_target {
                            1.0 - conf12
                        } else {
                            conf12
                        }
                    }
                    Verdict12::Tie => 0.5,
                };
                weighted_wr_terms.push(conf_target_better.clamp(0.0, 1.0));

                // Save categorical verdict
                let entry = results
                    .entry(id.to_string())
                    .or_insert_with(|| {
                        let mut m = JsonMap::new();
                        m.insert("prompt_count".to_string(), json!(id));
                        m
                    });
                entry.insert(key.clone(), serde_json::to_value(verdict)?);

                // LC stats collection
                let y = verdict.to_score();
                let len_diff =
                    (a_target.chars().count() as f64) - (a_ref.chars().count() as f64);
                lc_pairs_ols.push((y, len_diff));
                if let Verdict::Win | Verdict::Loss = verdict {
                    let y_bin: u8 = if let Verdict::Win = verdict { 1 } else { 0 };
                    glm_pairs.push((y_bin, len_diff));
                }

                if let Err(e) = save_results(&results, &out_path) {
                    logger.log(&format!("[WARN] Could not save partial results: {e}"));
                }
                if args.delay_ms > 0 {
                    sleep(Duration::from_millis(args.delay_ms)).await;
                }
            }
        }
        bar.finish();

        // Final save
        save_results(&results, &out_path)?;

        // Categorical summary
        logger.log("\n--- Per-target Overall W/L/T (target vs reference) ---");
        print_summary_report(&results, &mut logger);

        // Weighted Win Rate (confidence-weighted)
        if !weighted_wr_terms.is_empty() {
            let wwr = 100.0 * (weighted_wr_terms.iter().sum::<f64>() / weighted_wr_terms.len() as f64);
            logger.log(&format!("Weighted Win Rate (confidence-weighted): {:.2}%", wwr));
        } else {
            logger.log("Weighted Win Rate: n/a (no comparisons).");
        }

        // Robust summary (worst/best/avg/stdev across paraphrases per task)
        logger.log("\n--- RobustAlpacaEval Summary (per-task worst/best/avg/stdev; then macro-avg) ---");
        print_robust_summary(&results, &mut logger);

        // OLS length control (legacy)
        if args.length_control {
            logger.log("\n--- Length-Controlled (OLS) Win Rate (y ~ α + β·Δlen) ---");
            if lc_pairs_ols.len() >= 10 {
                let (alpha, beta) = fit_linear_ols(&lc_pairs_ols);
                let ys_adj: Vec<f64> = lc_pairs_ols
                    .iter()
                    .map(|(y, x)| (y - beta * x).clamp(0.0, 1.0))
                    .collect();
                let lc_win = mean(&ys_adj) * 100.0;
                logger.log(&format!(
                    "OLS α (pred. p@Δlen=0): {:.2}%   β (len effect): {:.6}",
                    alpha * 100.0,
                    beta
                ));
                logger.log(&format!(
                    "OLS Length-Controlled Win Rate: {:.2}%",
                    lc_win
                ));
            } else {
                logger.log("Not enough comparisons to fit OLS (need >=10). Skipping.");
            }
        }

        // After finishing this target, decrement the global cap and stop if exhausted.
        if remaining_records != usize::MAX {
            remaining_records = remaining_records.saturating_sub(selected_len);
            logger.log(&format!(
                "Global record cap remaining after this target: {}",
                remaining_records
            ));
            if remaining_records == 0 {
                logger.log("Global record cap exhausted – stopping.");
                break; // break out of the `for target_path in &args.answers_targets` loop
            }
        }

        // GLM / Logit length control (additional, closer to AlpacaEval 2.0 spirit)
        logger.log("\n--- Length-Controlled (GLM-Logit) Win Rate (logit(p)=α+β·Δlen) ---");
        let glm_pairs_clean: Vec<(f64, f64)> = glm_pairs
            .iter()
            .map(|(y, x)| (*y as f64, *x))
            .collect();
        if glm_pairs_clean.len() >= 10 {
            let (alpha, beta) = fit_logit_1d(&glm_pairs_clean);
            let p0 = 1.0 / (1.0 + (-alpha).exp()); // predicted p(target>ref) at Δlen=0
            logger.log(&format!(
                "Logit α (p@Δlen=0): {:.2}%   β (len effect): {:.6}",
                p0 * 100.0,
                beta
            ));
        } else {
            logger.log("Not enough non-tie comparisons to fit GLM (need >=10). Skipping.");
        }
    }

    Ok(())
}

// Utilities
fn sanitize_for_fs(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '/' | '\\' | '"' | '<' | '>' | '|' | ':' | '*' | '?' => '_',
            _ => c,
        })
        .collect()
}

fn load_existing_results(
    path: &Path,
    logger: &mut Logger,
) -> Result<HashMap<String, JsonMap<String, Value>>> {
    if path.exists() {
        logger.log(&format!("Loading existing output for resume: {}", path.display()));
        let content = fs::read_to_string(path)?;
        if content.trim().is_empty() {
            logger.log("Output file empty. Starting fresh.");
            return Ok(HashMap::new());
        }
        let items: Vec<JsonMap<String, Value>> = serde_json::from_str(&content)
            .with_context(|| format!("Could not parse existing results: {}", path.display()))?;
        items
            .into_iter()
            .map(|obj| {
                let id = obj
                    .get("prompt_count")
                    .and_then(|v| v.as_u64().or(v.as_str().and_then(|s| s.parse().ok())))
                    .ok_or_else(|| anyhow!("Missing/invalid prompt_count in existing output"))?
                    .to_string();
                Ok((id, obj))
            })
            .collect::<Result<_, _>>()
    } else {
        Ok(HashMap::new())
    }
}

fn save_results(
    results: &HashMap<String, JsonMap<String, Value>>,
    output_path: &Path,
) -> Result<()> {
    let mut vec_out: Vec<JsonMap<String, Value>> = results.values().cloned().collect();
    vec_out.sort_by(|a, b| {
        let ka = a.get("prompt_count").and_then(Value::as_u64).unwrap_or(0);
        let kb = b.get("prompt_count").and_then(Value::as_u64).unwrap_or(0);
        ka.cmp(&kb)
    });
    let json_string = serde_json::to_string_pretty(&vec_out)?;
    if let Some(p) = output_path.parent() {
        fs::create_dir_all(p)?;
    }
    fs::write(output_path, json_string)?;
    Ok(())
}

fn print_summary_report(results: &HashMap<String, JsonMap<String, Value>>, logger: &mut Logger) {
    let mut overall_counts = BTreeMap::new();
    let mut by_type_counts: BTreeMap<String, BTreeMap<Verdict, u32>> = BTreeMap::new();

    for item in results.values() {
        for (key, value) in item {
            if key == "prompt_count" {
                continue;
            }
            let verdict_result: Result<Verdict, _> = serde_json::from_value(value.clone());
            if let Ok(v) = verdict_result {
                *overall_counts.entry(v).or_insert(0) += 1;
                *by_type_counts
                    .entry(key.clone())
                    .or_default()
                    .entry(v)
                    .or_insert(0) += 1;
            } else if let Some(s) = value.as_str() {
                let v = match s.to_lowercase().as_str() {
                    "win" => Some(Verdict::Win),
                    "loss" => Some(Verdict::Loss),
                    "tie" => Some(Verdict::Tie),
                    _ => None,
                };
                if let Some(vx) = v {
                    *overall_counts.entry(vx).or_insert(0) += 1;
                    *by_type_counts
                        .entry(key.clone())
                        .or_default()
                        .entry(vx)
                        .or_insert(0) += 1;
                }
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

    let win_rate = (wins as f64 + 0.5 * (ties as f64)) / (total as f64) * 100.0;
    logger.log("\n--- Overall Results ---");
    logger.log(&format!("Total Comparisons: {}", total));
    logger.log(&format!("Win Rate (target vs reference): {:.2}%", win_rate));
    logger.log(&format!(
        "  - Wins: {} ({:.2}%)",
        wins,
        (wins as f64) / (total as f64) * 100.0
    ));
    logger.log(&format!(
        "  - Ties: {} ({:.2}%)",
        ties,
        (ties as f64) / (total as f64) * 100.0
    ));
    logger.log(&format!(
        "  - Losses: {} ({:.2}%)",
        losses,
        (losses as f64) / (total as f64) * 100.0
    ));

    logger.log("\n--- Win Rate by Paraphrase Type ---");
    let mut types: Vec<_> = by_type_counts.into_iter().collect();
    types.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));
    for (key, counts) in types {
        let n = counts.values().sum::<u32>() as f64;
        let wr = ((*counts.get(&Verdict::Win).unwrap_or(&0) as f64)
            + 0.5 * (*counts.get(&Verdict::Tie).unwrap_or(&0) as f64))
            / n
            * 100.0;
        logger.log(&format!("- {:<30}: {:.2}% (n={})", key, wr, n as u32));
    }
}

fn print_robust_summary(
    results: &HashMap<String, JsonMap<String, Value>>,
    logger: &mut Logger,
) {
    // Per item (prompt_count): worst/best/avg/stdev across available paraphrases
    let mut per_item: Vec<(f64, f64, f64, f64, usize)> = Vec::new();
    for item in results.values() {
        let mut scores: Vec<f64> = Vec::new();
        for (key, value) in item {
            if key == "prompt_count" {
                continue;
            }
            let v: Option<Verdict> = serde_json::from_value(value.clone()).ok().or_else(|| {
                value.as_str().and_then(|s| match s.to_lowercase().as_str() {
                    "win" => Some(Verdict::Win),
                    "loss" => Some(Verdict::Loss),
                    "tie" => Some(Verdict::Tie),
                    _ => None,
                })
            });
            if let Some(vv) = v {
                scores.push(vv.to_score());
            }
        }
        if !scores.is_empty() {
            let worst = scores.iter().cloned().fold(1.0_f64, |a, b| a.min(b));
            let best = scores.iter().cloned().fold(0.0_f64, |a, b| a.max(b));
            let avg = mean(&scores);
            let sd = stdev(&scores, Some(avg));
            per_item.push((worst, best, avg, sd, scores.len()));
        }
    }

    if per_item.is_empty() {
        logger.log("No data for robust summary.");
        return;
    }

    let avg_worst = mean(&(per_item.iter().map(|t| t.0).collect::<Vec<_>>()));
    let avg_best = mean(&(per_item.iter().map(|t| t.1).collect::<Vec<_>>()));
    let avg_avg = mean(&(per_item.iter().map(|t| t.2).collect::<Vec<_>>()));
    let avg_sd = mean(&(per_item.iter().map(|t| t.3).collect::<Vec<_>>()));

    logger.log(&format!("Macro-Average across tasks (vs reference):"));
    logger.log(&format!(
        "  worst: {:.2}%   best: {:.2}%   avg: {:.2}%   stdev: {:.2}%",
        100.0 * avg_worst,
        100.0 * avg_best,
        100.0 * avg_avg,
        100.0 * avg_sd
    ));
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / (v.len() as f64)
    }
}
fn stdev(v: &[f64], pre: Option<f64>) -> f64 {
    if v.len() < 2 {
        0.0
    } else {
        let m = pre.unwrap_or_else(|| mean(v));
        let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / ((v.len() - 1) as f64);
        var.sqrt()
    }
}

// Simple OLS fit: y = alpha + beta * x
fn fit_linear_ols(pairs: &[(f64, f64)]) -> (f64, f64) {
    let n = pairs.len() as f64;
    let (sumx, sumy, sumxy, sumxx) =
        pairs
            .iter()
            .fold((0.0, 0.0, 0.0, 0.0), |acc, (y, x)| (acc.0 + x, acc.1 + y, acc.2 + x * y, acc.3 + x * x));
    let xbar = sumx / n;
    let ybar = sumy / n;
    let sxx = sumxx - n * xbar * xbar;
    let sxy = sumxy - n * xbar * ybar;
    let beta = if sxx.abs() < 1e-9 { 0.0 } else { sxy / sxx };
    let alpha = ybar - beta * xbar;
    (alpha, beta)
}

// One-covariate logistic regression via IRLS (ties excluded upstream)
fn fit_logit_1d(pairs: &[(f64, f64)]) -> (f64, f64) {
    if pairs.is_empty() {
        return (0.0, 0.0);
    }
    let mut alpha = 0.0_f64;
    let mut beta = 0.0_f64;
    for _iter in 0..25 {
        let mut g00 = 0.0;
        let mut g01 = 0.0;
        let mut g11 = 0.0;
        let mut z0 = 0.0;
        let mut z1 = 0.0;
        for (y, x) in pairs.iter().copied() {
            let z = alpha + beta * x;
            let p = 1.0 / (1.0 + (-z).exp());
            let w = p * (1.0 - p) + 1e-9;
            let r = y - p; // residual
            g00 += w;
            g01 += w * x;
            g11 += w * x * x;
            z0 += r;
            z1 += r * x;
        }
        let det = g00 * g11 - g01 * g01 + 1e-12;
        if det.abs() < 1e-12 {
            break;
        }
        let d_alpha = (g11 * z0 - g01 * z1) / det;
        let d_beta = (-g01 * z0 + g00 * z1) / det;
        alpha += d_alpha;
        beta += d_beta;
        if d_alpha.abs() < 1e-6 && d_beta.abs() < 1e-6 {
            break;
        }
    }
    (alpha, beta)
}

// Judge prompt + API
fn build_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Ok(reqwest::Client::builder().default_headers(headers).build()?)
}

/// Judge prompt: strict rubric, TF/Relevance priority, style/length neutrality.
/// Also asks for optional confidence in [0,1] for weighted WR.
fn build_judge_prompt(instruction: &str, answer_1: &str, answer_2: &str) -> String {
    format!(
r#"You are an expert evaluator. Compare two answers to the user's instruction.

Primary criterion (weight highest): Task Fulfilment / Relevance (TF).
Also consider: factual correctness, helpfulness, harmfulness/safety.
Ignore style, verbosity, and formatting unless they materially change usefulness.
Do NOT reward length by itself. Do NOT guess hidden preferences beyond the instruction.

[Instruction]
{instruction}

[Answer 1]
{answer_1}

[Answer 2]
{answer_2}

Decide which answer better fulfills the instruction overall. If they are essentially equal, choose Tie.

Respond with ONLY a single JSON object:
- "verdict": one of "1", "2", or "Tie"
- "confidence": a number in [0,1] indicating how strongly you prefer the chosen answer
    - If Tie, use 0.5
    - If choosing between 1 vs 2, 0.5 means barely better; 1.0 means overwhelmingly better

Example valid responses:
{{"verdict":"1","confidence":0.74}}
{{"verdict":"Tie","confidence":0.5}}
"#
    )
}

#[derive(Deserialize)]
struct JudgeResponse {
    verdict: String,
    #[serde(default)]
    confidence: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Verdict12 {
    One,
    Two,
    Tie,
}

async fn query_gemini_for_judgment(
    client: &reqwest::Client,
    key: &str,
    model: &str,
    prompt: &str,
    temperature: f32,
) -> Result<(Verdict12, f64)> {
    let url = format!("{ENDPOINT}/models/{model}:generateContent?key={key}");
    let body = json!({
        "contents": [{ "role": "user", "parts": [{ "text": prompt }] }],
        "generationConfig": { "responseMimeType": "application/json", "temperature": temperature }
    });

    let resp = client.post(&url).json(&body).send().await?;
    let status = resp.status();
    let resp_text = resp.text().await?;
    if !status.is_success() {
        return Err(anyhow!("API Error {}: {}", status, resp_text));
    }

    let resp_json: Value = serde_json::from_str(&resp_text)
        .with_context(|| format!("Failed to parse judge response shell as JSON: {resp_text}"))?;
    let json_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| anyhow!("Unexpected response structure: `text` field not found"))?;

    let judge_response: JudgeResponse =
        serde_json::from_str(json_text.trim()).with_context(|| {
            format!("Failed to parse inner judge JSON: {json_text}")
        })?;

    let v = match judge_response.verdict.trim() {
        "1" | "One" | "A" | "a" => Verdict12::One,
        "2" | "Two" | "B" | "b" => Verdict12::Two,
        s if s.eq_ignore_ascii_case("tie") => Verdict12::Tie,
        other => return Err(anyhow!("Invalid verdict from judge: {}", other)),
    };
    let conf = judge_response.confidence.unwrap_or(0.5).clamp(0.0, 1.0);
    Ok((v, conf))
}

// API key pool with strict rotation/exhaustion
// Replace your ApiKeyPool with this version:

struct ApiKeyPool {
    keys: Vec<String>,
    max_per: u32,
    usage: Vec<u32>,
    alive: Vec<bool>,
    idx: usize,
}
impl ApiKeyPool {
    fn new(mut keys: Vec<String>, max_per: u32) -> Result<Self> {
        if keys.is_empty() {
            if let Ok(k) = std::env::var("GOOGLE_API_KEY") { keys.push(k); }
        }
        if keys.is_empty() {
            return Err(anyhow!("Provide at least one --api-key or set GOOGLE_API_KEY"));
        }
        let n = keys.len();
        Ok(Self {
            usage: vec![0; n],
            alive: vec![true; n],
            keys,
            max_per,
            idx: 0,
        })
    }

    fn exhausted(&self) -> bool {
        !self.alive.iter().any(|&b| b)
    }

    fn advance_to_next_alive(&mut self) {
        if self.exhausted() { return; }
        let n = self.keys.len();
        for step in 1..=n {
            let j = (self.idx + step) % n;
            if self.alive[j] {
                self.idx = j;
                break;
            }
        }
    }

    /// Reserve a key for exactly ONE API call attempt (success OR failure).
    /// Returns Err if all keys are exhausted.
    fn take_key_for_call(&mut self) -> Result<String> {
        if self.exhausted() { return Err(anyhow!("All API keys exhausted")); }
        if !self.alive[self.idx] { self.advance_to_next_alive(); }
        if self.exhausted() { return Err(anyhow!("All API keys exhausted")); }

        let key = self.keys[self.idx].clone(); // <— OWN the string (no borrow)

        self.usage[self.idx] += 1;
        if self.usage[self.idx] >= self.max_per {
            self.alive[self.idx] = false;
            self.advance_to_next_alive();
        }
        Ok(key)
    }
}
