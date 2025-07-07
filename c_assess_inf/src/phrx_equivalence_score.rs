/*
cargo run --release -- \
  data/prompts_with_paraphrases.json \
  output/paraphrase_scores.json \
  --model "gemini-1.5-flash-latest" \
  --api-key $GOOGLE_API_KEY
*/

use anyhow::{anyhow, Context, Result};
use chrono::Local;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::HeaderMap;
use serde::Deserialize;
use serde_json::{json, Map as JsonMap, Value};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};
//use tiktoken_rs::CoreBPE;
use tokio::time::sleep;

// Model Configuration & Tokenizer
struct ModelLimits {
    input: usize,
    _output: usize,
}

fn get_model_limits(model_name: &str) -> ModelLimits {
    match model_name {
        "gemini-2.5-flash-preview-05-20" => ModelLimits { input: 1_048_576, _output: 65_536 },
        "gemini-2.5-flash-lite-preview-06-17" => ModelLimits { input: 1_000_000, _output: 64_000 },
        "gemini-2.5-flash" => ModelLimits { input: 1_048_576, _output: 65_536 },
        "gemini-2.5-pro" => ModelLimits { input: 1_048_576, _output: 65_536 },
        "gemini-2.0-flash" => ModelLimits { input: 1_048_576, _output: 8_192 },
        _ => ModelLimits { input: 1_000_000, _output: 8_192 },
    }
}

// Logger & Data Structures
struct Logger {
    writer: BufWriter<fs::File>,
}
impl Logger {
    fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        let file = fs::OpenOptions::new().create(true).append(true).write(true).open(p)?;
        Ok(Self { writer: BufWriter::new(file) })
    }
    fn log(&mut self, msg: &str) {
        let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(self.writer, "[{ts}] {msg}");
        let _ = self.writer.flush();
    }
}

#[derive(Debug, Deserialize, Clone)]
struct Record {
    prompt_count: u32,
    #[serde(alias = "instruction_original", alias = "instruction")]
    instruction_original: String,
    #[serde(flatten)]
    extra: JsonMap<String, Value>,
}

// Command Line Interface
#[derive(Parser, Debug)]
#[command(version, author, about = "Assess semantic fidelity of paraphrases with dynamic chunking and resume support.")]
struct Cli {
    prompts: PathBuf,
    output: PathBuf,
    #[arg(long, default_value = "gemini-1.5-flash-latest")]
    model: String,
    #[arg(long, default_value_t = 5)]
    max_attempts: u8,
    #[arg(long = "delay-ms", default_value_t = 200)]
    delay_ms: u64,
    #[arg(long = "api-call-maximum", default_value_t = 10000)]
    api_call_maximum: usize,
    #[arg(long = "api-key")]
    api_key: Option<String>,
}

// Core Functions (build_eval_prompt, query_gemini, etc. remain the same)
fn read_records<P: AsRef<Path>>(p: P, logger: &mut Logger) -> Result<Vec<Record>> {
    let raw = fs::read_to_string(&p).with_context(|| format!("cannot read {}", p.as_ref().display()))?;
    let records: Vec<Record> = serde_json::from_str(&raw).context("JSON parse error")?;
    if records.is_empty() {
        logger.log("[warn] Prompts file contained 0 records.");
    }
    Ok(records)
}

fn build_eval_prompt(original: &str, batch: &[(String, String)]) -> (String, String) {
    let instructions = String::from(
r#"You are an expert in linguistic semantics. Your task is to compare each provided "paraphrase" against the "Instruction original".
Your entire focus must be on the **semantic content of the request**. Ignore any differences in style, tone, politeness, or wording.
Score every single paraphrase **independently** using an **integer from 0 to 5**.
- **5 (Perfect paraphrase):** The paraphrase asks for the *exact same information or action* as the original. Nothing has been added or removed from the core request.
- **4 (Minor deviation):** The paraphrase asks for the same thing but adds a very small constraint or piece of information (e.g., "... and explain").
- **3 (Noticeable deviation):** The paraphrase clearly adds a new requirement (e.g., "format as a list") or omits a key part of the original request.
- **2 (Major deviation):** The core task is substantially different.
- **1 (Different request):** The paraphrase is on the same broad topic but asks for something completely different.
- **0 (Unrelated):** The paraphrase is nonsensical or irrelevant.
You MUST return a valid JSON object. The JSON object should contain **every single key** from the "Paraphrases to score" list. Do NOT add comments, explanations, or use markdown code fences.

Example Response Format:
{
  "instruct_aave": 5,
  "instruct_apologetic": 5,
  "instruct_comparison_table": 3
}

Instruction original:
"#);
    let mut paraphrases_text = String::from("\n\nParaphrases to score:\n");
    for (key, text) in batch {
        paraphrases_text.push_str(&format!("\"{}\": \"{}\"\n", key, text));
    }
    let full_original_text = format!("\"{}\"", original);
    let mut full_prompt = instructions.clone();
    full_prompt.push_str(&full_original_text);
    full_prompt.push_str(&paraphrases_text);
    (full_prompt, full_original_text)
}

async fn query_gemini(client: &reqwest::Client, key: &str, model: &str, prompt: String) -> Result<JsonMap<String, Value>> {
    let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}", model, key);
    let body = json!({
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": { "responseMimeType": "application/json", "temperature": 0.0, "topP": 0.95 },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    });
    let resp = client.post(url).json(&body).send().await?;
    if !resp.status().is_success() {
        return Err(anyhow!("{} — {}", resp.status(), resp.text().await?));
    }
    let raw: Value = resp.json().await?;
    let text = raw["candidates"][0]["content"]["parts"][0]["text"].as_str().ok_or_else(|| anyhow!("Unexpected response structure"))?;
    parse_response(text)
}

fn parse_response(s: &str) -> Result<JsonMap<String, Value>> {
    let cleaned = s.trim().trim_start_matches("```json").trim_start_matches("```").trim_end_matches("```").trim();
    if let Ok(v) = serde_json::from_str(cleaned) { return Ok(v); }
    if let (Some(start), Some(end)) = (cleaned.find('{'), cleaned.rfind('}')) {
        if let Ok(v) = serde_json::from_str(&cleaned[start..=end]) { return Ok(v); }
    }
    Err(anyhow!("Could not parse valid JSON from response: {}", s))
}


// Main Application Logic
#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let api_key = cli.api_key.or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .context("Missing Google API key. Provide it with --api-key or $GOOGLE_API_KEY")?;
    
    fs::create_dir_all("logs")?;
    let stem = cli.output.file_stem().expect("Output must have a file name");
    let ts = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let filename = format!("{}_{ts}.log", stem.to_string_lossy());
    let log_path = PathBuf::from("logs").join(filename);
    let mut logger = Logger::new(&log_path)?;
    logger.log(&format!("Script started. Model: {}", cli.model));

    let headers = HeaderMap::new();
    let client = reqwest::Client::builder().default_headers(headers).timeout(Duration::from_secs(180)).build()?;
    let bpe = tiktoken_rs::p50k_base().unwrap();
    let model_limits = get_model_limits(&cli.model);
    let effective_token_limit = (model_limits.input as f64 * 0.8) as usize;
    let input_records = read_records(&cli.prompts, &mut logger)?;

    // Load existing results into a HashMap for efficient lookup
    let mut results_map: HashMap<u32, JsonMap<String, Value>> = if cli.output.exists() {
        let file = fs::File::open(&cli.output)?;
        let existing_vec: Vec<JsonMap<String, Value>> = serde_json::from_reader(file).unwrap_or_default();
        logger.log(&format!("Loaded {} existing results from output file.", existing_vec.len()));
        existing_vec.into_iter().filter_map(|entry| {
            entry["prompt_count"].as_u64().map(|id| (id as u32, entry))
        }).collect()
    } else {
        HashMap::new()
    };

    let pb = ProgressBar::new(input_records.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) - ID {msg}")?
        .progress_chars("=>-"));

    let mut api_calls_made = 0;
    let mut all_errors: HashMap<u32, Vec<String>> = HashMap::new();

    'outer: for record in input_records {
        pb.set_message(format!("{}", record.prompt_count));
        let prompt_id = record.prompt_count;

        // Granular check for unscored items
        let all_paraphrases_in_input: Vec<_> = record.extra.iter()
            .filter_map(|(k, v)| if k.starts_with("instruct_") { v.as_str().map(|s| (k.clone(), s.to_string())) } else { None })
            .collect();

        let scored_keys: HashSet<String> = results_map.get(&prompt_id)
            .and_then(|entry| entry.get("scores"))
            .and_then(|scores| scores.as_object())
            .map(|scores_map| scores_map.keys().cloned().collect())
            .unwrap_or_default();

        let mut paraphrases_to_process: Vec<_> = all_paraphrases_in_input.into_iter()
            .filter(|(key, _)| !scored_keys.contains(key))
            .collect();
        
        if paraphrases_to_process.is_empty() {
            pb.inc(1);
            continue; // Skip if all paraphrases for this ID are already scored
        }
        logger.log(&format!("[info] ID {}: Found {} unscored paraphrases to process.", prompt_id, paraphrases_to_process.len()));
            
        let mut new_scores_for_this_id = JsonMap::new();

        // Dynamic Chunking Loop (now operates on the unscored subset)
        while !paraphrases_to_process.is_empty() {
            if api_calls_made >= cli.api_call_maximum { logger.log("[warn] API call maximum reached. Halting run."); break 'outer; }

            let (base_prompt_template, _) = build_eval_prompt(&record.instruction_original, &[]);
            let mut current_tokens = bpe.encode_with_special_tokens(&base_prompt_template).len();
            let mut chunk_paraphrases = Vec::new();
            
            let mut i = 0;
            while i < paraphrases_to_process.len() {
                let (key, text) = &paraphrases_to_process[i];
                let paraphrase_line = format!("\"{}\": \"{}\"\n", key, text);
                let paraphrase_tokens = bpe.encode_with_special_tokens(&paraphrase_line).len();
                if current_tokens + paraphrase_tokens > effective_token_limit { break; }
                current_tokens += paraphrase_tokens;
                chunk_paraphrases.push((key.clone(), text.clone()));
                i += 1;
            }

            if chunk_paraphrases.is_empty() && !paraphrases_to_process.is_empty() {
                let err_msg = format!("Paraphrase '{}' is too large to fit in a single API call.", paraphrases_to_process[0].0);
                logger.log(&format!("[error] ID {}: {}", prompt_id, &err_msg));
                all_errors.entry(prompt_id).or_default().push(err_msg);
                paraphrases_to_process.remove(0); continue;
            }
            paraphrases_to_process.drain(0..i);
            
            let (prompt, _) = build_eval_prompt(&record.instruction_original, &chunk_paraphrases);
            api_calls_made += 1;
            let mut success = false;
            
            for attempt in 1..=cli.max_attempts {
                logger.log(&format!("[info] ID {}: Calling API for chunk of {} paraphrases (attempt {}/{})", prompt_id, chunk_paraphrases.len(), attempt, cli.max_attempts));
                match query_gemini(&client, &api_key, &cli.model, prompt.clone()).await {
                    Ok(parsed_scores) => {
                        logger.log(&format!("[info] ID {}: API call SUCCEEDED on attempt {}", prompt_id, attempt));
                        new_scores_for_this_id.extend(parsed_scores);
                        success = true;
                        break;
                    }
                    Err(e) => {
                        logger.log(&format!("[error] ID {}: API call FAILED on attempt {}: {}", prompt_id, attempt, e));
                        if attempt < cli.max_attempts { sleep(Duration::from_secs(3 * attempt as u64)).await; }
                    }
                }
            }
            if !success {
                let err_msg = format!("Chunk of {} items failed after {} attempts.", chunk_paraphrases.len(), cli.max_attempts);
                logger.log(&format!("[fatal] ID {}: {}", prompt_id, &err_msg));
                all_errors.entry(prompt_id).or_default().push(err_msg);
            } else {
                sleep(Duration::from_millis(cli.delay_ms)).await;
            }
        }
        
        // Merge new scores into the results map and save
        if !new_scores_for_this_id.is_empty() {
            let entry = results_map.entry(prompt_id).or_insert_with(|| {
                json!({
                    "prompt_count": prompt_id,
                    "instruction_original": record.instruction_original,
                    "scores": {}
                }).as_object().unwrap().clone()
            });
            let scores = entry.get_mut("scores").unwrap().as_object_mut().unwrap();
            scores.extend(new_scores_for_this_id);
        }

        let mut final_results_vec: Vec<_> = results_map.values().cloned().collect();
        final_results_vec.sort_by_key(|e| e["prompt_count"].as_u64().unwrap_or(0));

        let mut writer = BufWriter::new(fs::File::create(&cli.output)?);
        serde_json::to_writer_pretty(&mut writer, &final_results_vec)?;
        writer.flush()?;

        pb.inc(1);
    }
    
    pb.finish_with_message("Processing complete");
    logger.log("RUN FINISHED");
    if all_errors.is_empty() {
        logger.log("No fatal errors were recorded during the run.");
    } else {
        logger.log(&format!("!!! Found fatal errors for {} prompt IDs:", all_errors.len()));
        for (id, errors) in &all_errors {
            logger.log(&format!("  - ID {}:", id));
            for err in errors { logger.log(&format!("    - {}", err)); }
        }
    }
    Ok(())
}
