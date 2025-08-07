/*
OPENAI WAY OF GENERATING ANSWERS

generate LLM answers for every paraphrase in a prompts file using the OpenAI API
set key either with the flag or as an environment variable: export OPENAI_API_KEY="sk-..

cargo robust_alpaca_genbasline \
    --prompts a_data/alpaca/50k_phrxed.json \
    --output c_assess_inf/output50k/gpt4_answers_1440.json \
    --model gpt-4o \
    --api-key "sk-proj-T-xxx-xxx" \
    --log-name "GPT4_Baseline_Gen_1440" \
    --api-call-max 1440 \
    >> logs/robalev_gpt4_gen_1440_$(date +%F_%T).log 2>&1 &
*/

use anyhow::{anyhow, Context, Result};
use chrono::Local;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize};
use serde_json::{json, Map as JsonMap, Value};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::time::sleep;

const OPENAI_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";

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
        let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(self.writer, "[{ts}] {msg}");
        println!("[{ts}] {msg}");
        let _ = self.writer.flush();
    }
}

// Serde Helpers
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
    #[serde(flatten)]
    instructions: JsonMap<String, Value>,
    #[serde(default)]
    input: String,
}

// CLI Definition
#[derive(Parser, Debug)]
#[command(version, about = "Generates LLM answers for all paraphrases in a prompt file using the OpenAI API.")]
struct Cli {
    // Path to the JSON file containing prompts and their paraphrases
    #[arg(long)]
    prompts: PathBuf,
    // Path to write the output JSON file with generated answers
    #[arg(long)]
    output: PathBuf,
    // The OpenAI model to use for generation (e.g., gpt-4o, gpt-4-turbo, gpt-3.5-turbo)
    #[arg(long, default_value = "gpt-4o")]
    model: String,
    // OpenAI API key- Can also be set via the OPENAI_API_KEY environment variable
    #[arg(long)]
    api_key: Option<String>,
    // A name for the run, used in the log file name
    #[arg(long, default_value = "AnswerGen")]
    log_name: String,
    // System prompt to guide the model's behavior
    #[arg(long, default_value = "You are a helpful assistant. Provide a direct and concise answer to the user's request.")]
    system_prompt: String,
    // Maximum number of API call attempts for each prompt
    #[arg(long, default_value_t = 3)]
    max_attempts: u8,
    // Delay in milliseconds between API calls
    #[arg(long, default_value_t = 200)]
    delay_ms: u64,
    // Stop after this many API calls
    #[arg(long, default_value_t = 10_000)]
    api_call_max: u32,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    fs::create_dir_all("logs")?;
    let ts = Local::now().format("%Y%m%d-%H%M%S");
    let log_path = Path::new("logs").join(format!("{}_{}.log", cli.log_name, ts));
    let mut logger = Logger::new(&log_path)?;
    logger.log(&format!("Run started – model={}", cli.model));

    let api_key = cli.api_key.or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .context("Provide --api-key or set OPENAI_API_KEY environment variable")?;

    logger.log(&format!("Reading prompts from: {}", cli.prompts.display()));
    let prompt_records: Vec<PromptRecord> = read_records(&cli.prompts, &mut logger)?;

    let mut answers: HashMap<String, JsonMap<String, Value>> = load_existing_answers(&cli.output, &mut logger)?;
    
    let client = build_openai_client(&api_key)?;
    let bar = ProgressBar::new(prompt_records.len() as u64)
        .with_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})").unwrap());

    let mut api_calls_used = 0u32;
    'outer: for prompt_rec in &prompt_records {
        bar.inc(1);
        let id_str = prompt_rec.prompt_count.to_string();

        let keys_to_process: Vec<_> = prompt_rec.instructions.keys()
            .filter(|k| k.starts_with("instruct_") || k == &"instruction_original")
            .cloned()
            .collect();
        
        let already_done_keys: HashSet<String> = answers.get(&id_str)
            .map(|obj| obj.keys().cloned().collect())
            .unwrap_or_default();
        
        for key in keys_to_process {
            if already_done_keys.contains(&key) { continue; }
            if api_calls_used >= cli.api_call_max {
                logger.log("API call limit reached -> aborting early.");
                break 'outer;
            }

            let Some(instruction_text) = prompt_rec.instructions.get(&key).and_then(Value::as_str) else {
                logger.log(&format!("ID {}: Key '{}' has non-string value, skipping.", id_str, key));
                continue;
            };

            let user_prompt = if prompt_rec.input.is_empty() {
                instruction_text.to_string()
            } else {
                format!("{}\n\n[Input Data]\n{}", instruction_text, prompt_rec.input)
            };

            let mut answer = String::new();
            for attempt in 1..=cli.max_attempts {
                match query_openai(&client, &cli.model, &cli.system_prompt, &user_prompt).await {
                    Ok(generated_text) => {
                        answer = generated_text;
                        break;
                    }
                    Err(e) if attempt < cli.max_attempts => {
                        logger.log(&format!("ID {} key {}: API attempt {} failed: {}. Retrying...", id_str, key, attempt, e));
                        let backoff = 500u64 * 2u64.pow(attempt as u32);
                        sleep(Duration::from_millis(backoff)).await;
                    }
                    Err(e) => {
                        logger.log(&format!("ID {} key {}: All API attempts failed: {}", id_str, key, e));
                        answer = format!("[GENERATION_ERROR: {}]", e); // Mark as failed in output
                        break;
                    }
                }
            }
            api_calls_used += 1;

            let entry = answers.entry(id_str.clone()).or_insert_with(|| {
                let mut map = JsonMap::new();
                map.insert("prompt_count".into(), json!(prompt_rec.prompt_count));
                if !prompt_rec.input.is_empty() {
                    map.insert("input".into(), json!(prompt_rec.input));
                }
                map
            });
            entry.insert(key.clone(), json!(answer));

            if let Err(e) = save_answers(&answers, &cli.output) {
                logger.log(&format!("[ERROR] Failed to save intermediate results: {}", e));
            }
            if cli.delay_ms > 0 { sleep(Duration::from_millis(cli.delay_ms)).await; }
        }
    }
    bar.finish();

    logger.log(&format!("Finished. Writing final results to {}", cli.output.display()));
    save_answers(&answers, &cli.output)?;

    Ok(())
}

fn read_records<T: for<'de> Deserialize<'de>>(path: &Path, logger: &mut Logger) -> Result<Vec<T>> {
    let content = fs::read_to_string(path).with_context(|| {
        let msg = format!("[FATAL] Could not read {}: {}", path.display(), path.display());
        logger.log(&msg);
        msg
    })?;
    serde_json::from_str(&content).with_context(|| {
        let msg = format!("[FATAL] JSON parse error in {}", path.display());
        logger.log(&msg);
        msg
    })
}

fn load_existing_answers(path: &Path, logger: &mut Logger) -> Result<HashMap<String, JsonMap<String, Value>>> {
    if !path.exists() {
        logger.log("No existing output file found. Starting fresh.");
        return Ok(HashMap::new());
    }
    
    logger.log(&format!("Loading existing output for resume mode: {}", path.display()));
    let content = fs::read_to_string(path)?;
    if content.trim().is_empty() {
        logger.log("Output file is empty. Starting fresh.");
        return Ok(HashMap::new());
    }

    let items: Vec<JsonMap<String, Value>> = serde_json::from_str(&content)
        .context("Could not parse existing answers file as JSON array")?;
    
    items.into_iter().map(|obj| {
        let id = obj.get("prompt_count")
            .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
            .ok_or_else(|| anyhow!("Missing or invalid prompt_count in existing output"))?
            .to_string();
        Ok((id, obj))
    }).collect()
}

fn save_answers(answers: &HashMap<String, JsonMap<String, Value>>, output_path: &Path) -> Result<()> {
    let mut vec_out: Vec<JsonMap<String, Value>> = answers.values().cloned().collect();
    vec_out.sort_by_key(|m| m.get("prompt_count").and_then(Value::as_u64).unwrap_or(0));
    let json_string = serde_json::to_string_pretty(&vec_out)?;
    if let Some(p) = output_path.parent() { fs::create_dir_all(p)?; }
    fs::write(output_path, json_string)?;
    Ok(())
}

fn build_openai_client(api_key: &str) -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key))?);
    Ok(reqwest::Client::builder().default_headers(headers).build()?)
}

async fn query_openai(
    client: &reqwest::Client,
    model: &str,
    system_prompt: &str,
    user_prompt: &str,
) -> Result<String> {
    let body = json!({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
    });

    let resp = client.post(OPENAI_ENDPOINT).json(&body).send().await?;
    let status = resp.status();
    let resp_text = resp.text().await?;

    if !status.is_success() {
        return Err(anyhow!("OpenAI API Error {}: {}", status, resp_text));
    }

    let resp_json: Value = serde_json::from_str(&resp_text)
        .with_context(|| format!("Failed to parse OpenAI response as JSON: {}", resp_text))?;

    let content = resp_json["choices"][0]["message"]["content"].as_str()
        .ok_or_else(|| anyhow!("Could not find answer content in OpenAI response: {}", resp_text))?;

    Ok(content.to_string())
}
