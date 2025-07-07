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
//use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest::header::HeaderMap;
use serde::Deserialize;
use serde_json::{json, Map as JsonMap, Value};
use std::{
    collections::HashMap,
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
    _output: usize, // Reserved for future use
}

// Returns the hard-coded token limits for a given model name
fn get_model_limits(model_name: &str) -> ModelLimits {
    match model_name {
        "gemini-2.5-flash-preview-05-20" => ModelLimits { input: 1_048_576, _output: 65_536 },
        "gemini-2.5-flash-lite-preview-06-17" => ModelLimits { input: 1_000_000, _output: 64_000 },
        "gemini-2.5-flash" => ModelLimits { input: 1_048_576, _output: 65_536 },
        "gemini-2.5-pro" => ModelLimits { input: 1_048_576, _output: 65_536 },
        "gemini-2.0-flash" => ModelLimits { input: 1_048_576, _output: 8_192 },
        // Fallback for any other model, including "gemini-1.5-flash-latest"
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

#[derive(Debug, Deserialize)]
struct Record {
    prompt_count: u32,
    #[serde(alias = "instruction_original", alias = "instruction")]
    instruction_original: String,
    #[serde(flatten)]
    extra: JsonMap<String, Value>,
}

// Command Line Interface

#[derive(Parser, Debug)]
#[command(version, author, about = "Assess semantic fidelity of paraphrases with dynamic chunking.")]
struct Cli {
    // JSON file containing the prompts + paraphrases
    prompts: PathBuf,
    output: PathBuf,
    // Gemini model name
    #[arg(long, default_value = "gemini-1.5-flash-latest")]
    model: String,
    // Maximum attempts per failed request
    #[arg(long, default_value_t = 5)]
    max_attempts: u8,
    // Delay (ms) after every *successful* call
    #[arg(long = "delay-ms", default_value_t = 200)]
    delay_ms: u64,
    // Global cap on total calls for one run (progress is saved)
    #[arg(long = "api-call-maximum", default_value_t = 10000)]
    api_call_maximum: usize,
    // Google API key (alternatively: $GOOGLE_API_KEY)
    #[arg(long = "api-key")]
    api_key: Option<String>,
}

// Main Functions

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
- **5 (Perfect paraphrase):** The paraphrase asks for the *exact same information or action* as the original. Nothing has been added or removed from the core request. The wording can be completely different.
- **4 (Minor deviation):** The paraphrase asks for the same thing but adds a very small constraint or piece of information (e.g., "... and explain"). Or, it omits a similarly minor detail.
- **3 (Noticeable deviation):** The paraphrase clearly adds a new requirement (e.g., "explain your reasoning," "format as a list") or omits a key part of the original request. The resulting output would need to be different to be correct.
- **2 (Major deviation):** The core task is substantially different. While on the same topic, the instructions have been fundamentally changed.
- **1 (Different request):** The paraphrase is on the same broad topic but asks for something completely different.
- **0 (Unrelated):** The paraphrase is nonsensical, irrelevant, or fails to make a request.
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
    let stem = cli
        .output
        .file_stem()
        .expect("Output must have a file name");
    let ts = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let filename = format!("{}_{ts}.log", stem.to_string_lossy());
    let log_path = PathBuf::from("logs").join(filename);
    //let log_path = PathBuf::from("logs").join(stem).with_extension("log");
    let mut logger = Logger::new(&log_path)?;
    logger.log(&format!("Script started. Model: {}", cli.model));

    let headers = HeaderMap::new();
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .timeout(Duration::from_secs(180))
        .build()?;

    let bpe = tiktoken_rs::p50k_base().unwrap();
    let model_limits = get_model_limits(&cli.model);
    let token_safety_margin = 0.95; 
    let effective_token_limit = (model_limits.input as f64 * token_safety_margin) as usize;

    let records = read_records(&cli.prompts, &mut logger)?;
    
    let mut existing_results: Vec<JsonMap<String, Value>> = if cli.output.exists() {
        serde_json::from_reader(fs::File::open(&cli.output)?).unwrap_or_default()
    } else {
        Vec::new()
    };

    let pb = ProgressBar::new(records.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) - ID {msg}")?
        .progress_chars("=>-"));

    let mut api_calls_made = 0;
    let mut all_errors: HashMap<u32, Vec<String>> = HashMap::new();

    let already_scored = |results: &Vec<JsonMap<String, Value>>, id: u32| -> bool {
        results.iter()
            .any(|e| e.get("prompt_count")
                        .and_then(Value::as_u64)
                        .map(|v| v as u32 == id)
                        .unwrap_or(false))
    };

    'outer: for record in records {
        pb.set_message(format!("{}", record.prompt_count));
        if already_scored(&existing_results, record.prompt_count) {
            pb.inc(1);
            continue;
        }

        let mut paraphrases_to_process: Vec<_> = record.extra.iter()
            .filter_map(|(k, v)| if k.starts_with("instruct_") { v.as_str().map(|s| (k.clone(), s.to_string())) } else { None })
            .collect();
        
        let mut record_scores = JsonMap::new();
        
        // Flexible Chunking Loop
        while !paraphrases_to_process.is_empty() {
            if api_calls_made >= cli.api_call_maximum {
                logger.log("[warn] API call maximum reached. Halting run.");
                break 'outer;
            }

            //let (base_prompt_template, original_text) = build_eval_prompt(&record.instruction_original, &[]);
            let (base_prompt_template, _) = build_eval_prompt(&record.instruction_original, &[]);
            let mut current_tokens = bpe
                .encode_with_special_tokens(&base_prompt_template)
                .len();
            
            let mut chunk_paraphrases = Vec::new();
            
            // Greedily pack paraphrases into the current chunk
            let mut i = 0;
            while i < paraphrases_to_process.len() {
                let (key, text) = &paraphrases_to_process[i];
                let paraphrase_line = format!("\"{}\": \"{}\"\n", key, text);
                let paraphrase_tokens = bpe.encode_with_special_tokens(&paraphrase_line).len();

                if current_tokens + paraphrase_tokens > effective_token_limit {
                    break; // Chunk is full
                }
                
                current_tokens += paraphrase_tokens;
                chunk_paraphrases.push((key.clone(), text.clone()));
                i += 1;
            }

            if chunk_paraphrases.is_empty() && !paraphrases_to_process.is_empty() {
                let err_msg = format!("Paraphrase '{}' is too large to fit in a single API call.", paraphrases_to_process[0].0);
                logger.log(&format!("[error] ID {}: {}", record.prompt_count, &err_msg));
                all_errors.entry(record.prompt_count).or_default().push(err_msg);
                paraphrases_to_process.remove(0); // Skip this paraphrase
                continue;
            }
            
            paraphrases_to_process.drain(0..i); // Remove the processed items
            
            // API Call for the Chunk
            let (prompt, _) = build_eval_prompt(&record.instruction_original, &chunk_paraphrases);
            api_calls_made += 1;
            let mut success = false;
            
            for attempt in 1..=cli.max_attempts {
                logger.log(&format!("[info] ID {}: Calling API for chunk of {} paraphrases (attempt {}/{})", record.prompt_count, chunk_paraphrases.len(), attempt, cli.max_attempts));
                match query_gemini(&client, &api_key, &cli.model, prompt.clone()).await {
                    Ok(parsed_scores) => {
                        logger.log(&format!("[info] ID {}: API call SUCCEEDED on attempt {}", record.prompt_count, attempt));
                        record_scores.extend(parsed_scores);
                        success = true;
                        break;
                    }
                    Err(e) => {
                        logger.log(&format!("[error] ID {}: API call FAILED on attempt {}: {}", record.prompt_count, attempt, e));
                        if attempt < cli.max_attempts { sleep(Duration::from_secs(3 * attempt as u64)).await; }
                    }
                }
            }
            
            if !success {
                let err_msg = format!("Chunk of {} items failed after {} attempts.", chunk_paraphrases.len(), cli.max_attempts);
                logger.log(&format!("[fatal] ID {}: {}", record.prompt_count, &err_msg));
                all_errors.entry(record.prompt_count).or_default().push(err_msg);
            } else {
                sleep(Duration::from_millis(cli.delay_ms)).await;
            }
        }
        
        let mut final_entry = JsonMap::new();
        final_entry.insert("prompt_count".to_string(), json!(record.prompt_count));
        final_entry.insert("instruction_original".to_string(), json!(record.instruction_original));
        final_entry.insert("scores".to_string(), json!(record_scores));
        
        existing_results.push(final_entry);

        let mut writer = BufWriter::new(fs::File::create(&cli.output)?);
        serde_json::to_writer_pretty(&mut writer, &existing_results)?;
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
            for err in errors {
                logger.log(&format!("    - {}", err));
            }
        }
    }
    Ok(())
}
