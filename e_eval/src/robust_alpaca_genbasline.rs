/*
OPENAI WAY OF GENERATING ANSWERS

generate LLM answers for every paraphrase in a prompts file using the OpenAI API
set key either with the flag or as an environment variable: export OPENAI_API_KEY="sk-..

cargo robust_alpaca_genbasline \
    --prompts a_data/alpaca/paraphrases_500.json \
    --whitelist h_rae/data/whitelists/hinf_alta4_alpaca_inferences.json \
    --output h_rae/output/answer_baseline/gpt5_answers_alt4whitelist.json \
    --api-key "xxxxxxxxx" \
    --log-name "gpt5_answers_alt4whitelist" \
    --api-call-max 16400 \
    >> logs/robalev_gpt5_answers_alt4whitelist_$(date +%F_%T).log 2>&1 &


generate LLM answers for every paraphrase in a prompts file using the OpenAI API
set key either with the flag or as an environment variable: export OPENAI_API_KEY="sk-..

cargo robust_alpaca_genbasline \
    --prompts a_data/alpaca/50k_phrxed.json \
    --whitelist e_eval/output_robust_alpaca_eval/li9k_a1_notarg_whitelist.json \
    --output c_assess_inf/output50k/gpt4_answers_1440.json \
    --model gpt-4o \
    --api-key "xxxxxxxxxxxxxx" \
    --log-name "GPT4_Baseline_Gen_1548whitelist" \
    --api-call-max 1548 \
    >> logs/robalev_gpt4_gen_1548whitelist_$(date +%F_%T).log 2>&1 &
*/
use anyhow::{anyhow, Context, Result};
use chrono::Local;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::Deserialize;
use serde_json::{json, Map as JsonMap, Value};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::time::sleep;

// Tokenizer for token counting (add `tiktoken-rs = "0.5"` to Cargo.toml)
use tiktoken_rs::cl100k_base;

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
    #[arg(long)]
    prompts: PathBuf,
    #[arg(long)]
    output: PathBuf,
    // Path to an optional JSON file containing a list of prompt_count IDs to process
    #[arg(long)]
    whitelist: Option<PathBuf>,
    // Default to the cheapest model
    #[arg(long, default_value = "gpt-5-nano")]
    model: String,
    // OpenAI API key- Can also be set via the OPENAI_API_KEY environment variable
    #[arg(long)]
    api_key: Option<String>,
    // A name for the run, used in the log file name
    #[arg(long, default_value = "AnswerGen")]
    log_name: String,
    #[arg(long, default_value = "You are a helpful assistant. Provide a direct and concise answer to the user's request.")]
    system_prompt: String,
    // Maximum number of API call attempts for each prompt
    #[arg(long, default_value_t = 3)]
    max_attempts: u8,
    // Delay in milliseconds between API calls
    #[arg(long, default_value_t = 200)]
    delay_ms: u64,
    #[arg(long, default_value_t = 10_000)]
    api_call_max: u32,

    // Cap on input tokens for instruction value (skip if exceeded)
    #[arg(long, default_value_t = 120)]
    max_input_tokens: usize,

    // Pricing (USD per 1M tokens), defaults for GPT-5 nano
    #[arg(long, default_value_t = 0.05)]
    price_per_million_input: f64,
    #[arg(long, default_value_t = 0.40)]
    price_per_million_output: f64,

    // OPTIONAL temperature; if omitted, not sent (avoids GPT-5 nano 400)
    #[arg(long)]
    temperature: Option<f64>,

    // Hard cap on model output tokens (completion). 0 = no cap.
    #[arg(long, default_value_t = 256)]
    max_output_tokens: u32,

    // Reasoning effort for GPT-5 series: minimal | low | medium | high
    #[arg(long, default_value = "minimal")]
    reasoning_effort: String,

    // fallback when reasoning starves output
    #[arg(long, default_value_t = true)]
    fallback_on_empty: bool,

    // which model to fallback to if we detect reasoning starvation
    #[arg(long, default_value = "gpt-5-chat-latest")]
    fallback_model: String,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    fs::create_dir_all("logs")?;
    let ts = Local::now().format("%Y%m%d-%H%M%S");
    let log_path = Path::new("logs").join(format!("{}_{}.log", cli.log_name, ts));
    let mut logger = Logger::new(&log_path)?;
    logger.log(&format!("Run started – model={}", cli.model));

    let api_key = cli
        .api_key
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .context("Provide --api-key or set OPENAI_API_KEY environment variable")?;

    let whitelist_set: Option<HashSet<String>> = if let Some(path) = &cli.whitelist {
        logger.log(&format!("Reading whitelist from: {}", path.display()));
        let json_values: Vec<Value> =
            read_records(path, &mut logger).context("Failed to read or parse whitelist file")?;
        let ids: HashSet<String> = json_values
            .into_iter()
            .filter_map(|v| match v {
                Value::Number(n) => n.as_u64().map(|i| i.to_string()),
                Value::String(s) => Some(s),
                _ => {
                    logger.log(&format!(
                        "[WARN] Ignoring non-string/non-number value in whitelist: {:?}",
                        v
                    ));
                    None
                }
            })
            .collect();
        if ids.is_empty() {
            logger.log("[WARN] Whitelist was provided but resulted in an empty set of IDs.");
        }
        Some(ids)
    } else {
        None
    };

    logger.log(&format!("Reading prompts from: {}", cli.prompts.display()));
    let all_prompt_records: Vec<PromptRecord> = read_records(&cli.prompts, &mut logger)?;

    let prompt_records = if let Some(set) = whitelist_set {
        let original_count = all_prompt_records.len();
        let filtered: Vec<PromptRecord> = all_prompt_records
            .into_iter()
            .filter(|rec| set.contains(&rec.prompt_count.to_string()))
            .collect();
        logger.log(&format!(
            "Whitelist applied: {} of {} records selected for processing.",
            filtered.len(),
            original_count
        ));
        filtered
    } else {
        all_prompt_records
    };

    let mut answers: HashMap<String, JsonMap<String, Value>> =
        load_existing_answers(&cli.output, &mut logger)?;

    let client = build_openai_client(&api_key)?;
    let bar = ProgressBar::new(prompt_records.len() as u64).with_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap(),
    );

    // Running usage + planned total
    let mut total_input_tokens: u64 = 0;
    let mut total_output_tokens: u64 = 0;
    let mut api_calls_used = 0u32;

    // Compute planned total calls (respect resume & api_call_max)
    let mut total_pending_keys: usize = 0;
    for rec in &prompt_records {
        let keys = rec
            .instructions
            .keys()
            .filter(|k| k.starts_with("instruct_") || k == &"instruction_original");
        let done = answers
            .get(&rec.prompt_count.to_string())
            .map(|obj| obj.keys().cloned().collect::<HashSet<_>>())
            .unwrap_or_default();
        total_pending_keys += keys.filter(|k| !done.contains(*k)).count();
    }
    let planned_total_calls = std::cmp::min(total_pending_keys as u32, cli.api_call_max);
    logger.log(&format!(
        "Planned total calls (bounded by api_call_max): {}",
        planned_total_calls
    ));

    'outer: for prompt_rec in &prompt_records {
        bar.inc(1);
        let id_str = prompt_rec.prompt_count.to_string();

        let keys_to_process: Vec<_> = prompt_rec
            .instructions
            .keys()
            .filter(|k| k.starts_with("instruct_") || k == &"instruction_original")
            .cloned()
            .collect();

        let already_done_keys: HashSet<String> = answers
            .get(&id_str)
            .map(|obj| obj.keys().cloned().collect())
            .unwrap_or_default();

        for key in keys_to_process {
            if already_done_keys.contains(&key) {
                continue;
            }
            if api_calls_used >= cli.api_call_max {
                logger.log("API call limit reached -> aborting early.");
                break 'outer;
            }

            let Some(instruction_text) = prompt_rec.instructions.get(&key).and_then(Value::as_str)
            else {
                logger.log(&format!(
                    "ID {}: Key '{}' has non-string value, skipping.",
                    id_str, key
                ));
                continue;
            };

            // Build the actual user prompt (may include input) AFTER length check
            let user_prompt = if prompt_rec.input.is_empty() {
                instruction_text.to_string()
            } else {
                format!("{}\n\n[Input Data]\n{}", instruction_text, prompt_rec.input)
            };

            // Token cap now applies to the FULL user content
            let instr_tokens = count_tokens(&user_prompt);
            if instr_tokens > cli.max_input_tokens {
                logger.log(&format!(
                    "ID {} key {}: Skipping – {} tokens exceeds cap {} (full user content)",
                    id_str, key, instr_tokens, cli.max_input_tokens
                ));
                // Do NOT insert this key into the output JSON
                continue;
            }

            let mut answer = String::new();
            let mut p_tok: u64 = 0;
            let mut c_tok: u64 = 0;

            for attempt in 1..=cli.max_attempts {
                match query_openai(
                    &client,
                    &cli.model,
                    &cli.system_prompt,
                    &user_prompt,
                    cli.temperature,
                    cli.max_output_tokens,
                    cli.fallback_on_empty,
                    &cli.fallback_model,
                    &cli.reasoning_effort,
                    &mut logger,
                    &id_str,
                    &key,
                )
                .await
                {
                    Ok((generated_text, prompt_tokens, completion_tokens)) => {
                        answer = generated_text;
                        p_tok = prompt_tokens;
                        c_tok = completion_tokens;

                        logger.log(&format!(
                            "ID {} key {}: saved {} chars ({} in / {} out tokens)",
                            id_str, key, answer.len(), p_tok, c_tok
                        ));
                        break;
                    }
                    Err(e) if attempt < cli.max_attempts => {
                        logger.log(&format!(
                            "ID {} key {}: API attempt {} failed: {}. Retrying...",
                            id_str, key, attempt, e
                        ));
                        let backoff = 500u64 * 2u64.pow(attempt as u32);
                        sleep(Duration::from_millis(backoff)).await;
                    }
                    Err(e) => {
                        logger.log(&format!(
                            "ID {} key {}: All API attempts failed: {}",
                            id_str, key, e
                        ));
                        answer = format!("[GENERATION_ERROR: {}]", e); // Mark as failed in output
                        break;
                    }
                }
            }
            api_calls_used += 1;

            // accumulate usage & log cost so far
            total_input_tokens += p_tok;
            total_output_tokens += c_tok;
            let cost_in =
                (total_input_tokens as f64 / 1_000_000f64) * cli.price_per_million_input;
            let cost_out =
                (total_output_tokens as f64 / 1_000_000f64) * cli.price_per_million_output;
            let cost_total = cost_in + cost_out;
            let pct = (api_calls_used as f64 / planned_total_calls.max(1) as f64) * 100.0;
            logger.log(&format!(
                "[PROGRESS] {}/{} ({:.1}%) • usage_in={} usage_out={} • est_cost=${:.4} (in=${:.4}, out=${:.4})",
                api_calls_used,
                planned_total_calls,
                pct,
                total_input_tokens,
                total_output_tokens,
                cost_total,
                cost_in,
                cost_out
            ));

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
            if cli.delay_ms > 0 {
                sleep(Duration::from_millis(cli.delay_ms)).await;
            }
        }
    }
    bar.finish();

    logger.log(&format!(
        "Finished. Writing final results to {}",
        cli.output.display()
    ));
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

fn load_existing_answers(
    path: &Path,
    logger: &mut Logger,
) -> Result<HashMap<String, JsonMap<String, Value>>> {
    if !path.exists() {
        logger.log("No existing output file found. Starting fresh.");
        return Ok(HashMap::new());
    }

    logger.log(&format!(
        "Loading existing output for resume mode: {}",
        path.display()
    ));
    let content = fs::read_to_string(path)?;
    if content.trim().is_empty() {
        logger.log("Output file is empty. Starting fresh.");
        return Ok(HashMap::new());
    }

    // First attempt: strict parse
    match serde_json::from_str::<Vec<JsonMap<String, Value>>>(&content) {
        Ok(items) => {
            return items
                .into_iter()
                .map(|obj| {
                    let id = obj
                        .get("prompt_count")
                        .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
                        .ok_or_else(|| anyhow!("Missing or invalid prompt_count in existing output"))?
                        .to_string();
                    Ok((id, obj))
                })
                .collect();
        }
        Err(e1) => {
            logger.log(&format!(
                "[WARN] Strict JSON parse failed ({}). Attempting to repair trailing commas...",
                e1
            ));
            let cleaned = remove_trailing_commas(&content);
            match serde_json::from_str::<Vec<JsonMap<String, Value>>>(&cleaned) {
                Ok(items) => {
                    logger.log("[WARN] Recovered existing output by removing trailing commas. Writing cleaned file back.");
                    // Overwrite the file with a pretty, valid JSON array so future runs are clean
                    let pretty = serde_json::to_string_pretty(&items).unwrap_or(cleaned);
                    fs::write(path, pretty)?;
                    return items
                        .into_iter()
                        .map(|obj| {
                            let id = obj
                                .get("prompt_count")
                                .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
                                .ok_or_else(|| anyhow!("Missing or invalid prompt_count in existing output"))?
                                .to_string();
                            Ok((id, obj))
                        })
                        .collect();
                }
                Err(e2) => {
                    logger.log(&format!(
                        "[WARN] Repair failed as well ({}). Backing up corrupt file and starting fresh.",
                        e2
                    ));
                    // Backup corrupt file
                    let backup = path.with_extension(format!(
                        "{}.corrupt-{}",
                        path.extension().and_then(|s| s.to_str()).unwrap_or("json"),
                        Local::now().format("%Y%m%d-%H%M%S")
                    ));
                    let _ = fs::rename(path, &backup);
                    logger.log(&format!(
                        "[WARN] Moved corrupt file to {}. A new output file will be created.",
                        backup.display()
                    ));
                    return Ok(HashMap::new());
                }
            }
        }
    }
}

fn save_answers(
    answers: &HashMap<String, JsonMap<String, Value>>,
    output_path: &Path,
) -> Result<()> {
    let mut vec_out: Vec<JsonMap<String, Value>> = answers.values().cloned().collect();
    vec_out.sort_by_key(|m| m.get("prompt_count").and_then(Value::as_u64).unwrap_or(0));
    let json_string = serde_json::to_string_pretty(&vec_out)?;
    if let Some(p) = output_path.parent() {
        fs::create_dir_all(p)?;
    }
    fs::write(output_path, json_string)?;
    Ok(())
}

fn build_openai_client(api_key: &str) -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {}", api_key))?);
    Ok(reqwest::Client::builder().default_headers(headers).build()?)
}

// Remove trailing commas before ']' or '}' so we can recover from slightly-invalid JSON.
fn remove_trailing_commas(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        let c = bytes[i] as char;

        if c == ',' {
            // Look ahead to the next non-whitespace char
            let mut j = i + 1;
            while j < bytes.len() {
                let cj = bytes[j] as char;
                if cj.is_whitespace() {
                    j += 1;
                    continue;
                }
                if cj == ']' || cj == '}' {
                    // Skip writing this comma (i.e., drop it)
                    i += 1;
                    continue; // continue outer while without pushing ','
                }
                break;
            }
        }

        out.push(c);
        i += 1;
    }

    out
}

fn _cap_param_for_model(model: &str) -> &'static str {
    // Heuristic: GPT-5 series & nano prefer max_completion_tokens; 4-series use max_tokens
    if model.starts_with("gpt-5") || model.contains("nano") {
        "max_completion_tokens"
    } else {
        "max_tokens"
    }
}

fn truncate_to_tokens(s: &str, cap: u32) -> String {
    if cap == 0 { return s.to_string(); }
    if let Ok(enc) = cl100k_base() {
        let ids = enc.encode_ordinary(s);
        if ids.len() <= cap as usize {
            s.to_string()
        } else {
            let truncated = &ids[..cap as usize];
            enc.decode(truncated.to_vec()).unwrap_or_else(|_| {
                // very rare; fall back to a crude char cut
                s.chars().take((cap as usize) * 4).collect()
            })
        }
    } else {
        // tokenizer unavailable; crude fallback
        s.split_whitespace().take(cap as usize).collect::<Vec<_>>().join(" ")
    }
}

// token counting helper (instruction value only)
fn count_tokens(s: &str) -> usize {
    if let Ok(enc) = cl100k_base() {
        enc.encode_ordinary(s).len()
    } else {
        // Fallback: rough proxy
        s.split_whitespace().count()
    }
}

// Return (content, prompt_tokens, completion_tokens)
async fn query_openai(
    client: &reqwest::Client,
    model: &str,
    system_prompt: &str,
    user_prompt: &str,
    temperature: Option<f64>,
    max_output_tokens: u32,
    fallback_on_empty: bool,
    fallback_model: &str,
    reasoning_effort: &str,
    logger: &mut Logger,
    id_str: &str,
    key: &str,
) -> Result<(String, u64, u64)> {
    fn cap_param_for_model(model: &str) -> &'static str {
        if model.starts_with("gpt-5") || model.contains("nano") {
            "max_completion_tokens"
        } else {
            "max_tokens"
        }
    }

    // robust content extraction across shapes
    fn extract_content(resp_json: &Value) -> Option<String> {
        // 1) Standard Chat Completions: string
        if let Some(s) = resp_json["choices"][0]["message"]["content"].as_str() {
            if !s.is_empty() { return Some(s.to_string()); }
        }

        // Chat Completions: array of parts (some 5-series shapes)
        if let Some(arr) = resp_json["choices"][0]["message"]["content"].as_array() {
            let mut buf = String::new();
            for part in arr {
                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                    buf.push_str(t);
                } else if let Some(t) = part.get("text").and_then(|v| v.get("value")).and_then(|v| v.as_str()) {
                    buf.push_str(t);
                } else if let Some(s) = part.as_str() {
                    buf.push_str(s);
                }
            }
            if !buf.trim().is_empty() { return Some(buf); }
        }

        // Legacy: choices[].text
        if let Some(s) = resp_json["choices"][0]["text"].as_str() {
            if !s.is_empty() { return Some(s.to_string()); }
        }

        // Responses-API-like top-level shape
        if let Some(out) = resp_json.get("output").and_then(|v| v.as_array()) {
            let mut buf = String::new();
            for item in out {
                if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                    for c in content {
                        if let Some(t) = c.get("text").and_then(|v| v.get("value")).and_then(|v| v.as_str()) {
                            buf.push_str(t);
                        } else if let Some(t) = c.get("text").and_then(|v| v.as_str()) {
                            buf.push_str(t);
                        }
                    }
                }
            }
            if !buf.trim().is_empty() { return Some(buf); }
        }

        // Structured refusal string, if present
        if let Some(r) = resp_json["choices"][0]["message"]["refusal"].as_str() {
            if !r.is_empty() { return Some(r.to_string()); }
        }

        None
    }

    // Build request body with optional temperature and required cap.
    let build_body = |which_model: &str, temp_opt: Option<f64>, cap_key: &str| -> Value {
        let mut body = json!({
            "model": which_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            "response_format": { "type": "text" }
        });
        if let Some(t) = temp_opt {
            if let Some(obj) = body.as_object_mut() {
                obj.insert("temperature".to_string(), json!(t));
            }
        }
        if let Some(obj) = body.as_object_mut() {
            // hard cap the whole completion budget (visible + reasoning)
            if max_output_tokens > 0 {
                obj.insert(cap_key.to_string(), json!(max_output_tokens));
            }
            // Dial down hidden reasoning on GPT-5 family
            if which_model.starts_with("gpt-5") {
                // Try "minimal"; if the deployment doesn't support it, use "low".
                obj.insert("reasoning".to_string(), json!({ "effort": reasoning_effort }));
            }
        }
        body
    };

    // Single-attempt executor (used for primary and fallback calls)
    async fn exec_once(
        client: &reqwest::Client,
        body: &Value,
        id_str: &str,
        key: &str,
        logger: &mut Logger,
    ) -> Result<(Value, String, u64, u64, Option<u64>, Option<String>)> {
        let resp = client.post(OPENAI_ENDPOINT).json(body).send().await?;
        let status = resp.status();
        let resp_text = resp.text().await?;

        if !status.is_success() {
            return Err(anyhow!("OpenAI API Error {}: {}", status, resp_text));
        }
        let resp_json: Value = serde_json::from_str(&resp_text)
            .with_context(|| format!("Failed to parse OpenAI response as JSON: {}", resp_text))?;
        let content = extract_content(&resp_json).unwrap_or_default();
        let usage = &resp_json["usage"];
        let prompt_tokens =
            usage["prompt_tokens"].as_u64()
            .or_else(|| usage["input_tokens"].as_u64())
            .unwrap_or(0);
        let completion_tokens =
            usage["completion_tokens"].as_u64()
            .or_else(|| usage["output_tokens"].as_u64())
            .unwrap_or(0);
        let reasoning_tokens = usage.get("completion_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(|v| v.as_u64());
        let finish_reason = resp_json["choices"][0]["finish_reason"].as_str().map(|s| s.to_string());

        // Helpful debug if empty
        if content.trim().is_empty() {
            let preview = resp_text.chars().take(1500).collect::<String>();
            let shape_hint = format!(
                "shape: content(str)={:?} content(arr)={:?} choices.text(str)={:?} top.output(arr)={:?}",
                resp_json["choices"][0]["message"]["content"].as_str().is_some(),
                resp_json["choices"][0]["message"]["content"].as_array().is_some(),
                resp_json["choices"][0]["text"].as_str().is_some(),
                resp_json.get("output").and_then(|v| v.as_array()).is_some()
            );
            logger.log(&format!(
                "ID {} key {}: Empty content despite 200 OK. {} Raw (<=1500c): {}",
                id_str, key, shape_hint, preview
            ));
        }

        Ok((resp_json, content, prompt_tokens, completion_tokens, reasoning_tokens, finish_reason))
    }

    // Primary call
    let mut cap_param = cap_param_for_model(model); // "max_tokens" or "max_completion_tokens"
    let mut body = build_body(model, temperature, cap_param);
    let (resp_json, content, p_tok, c_tok, reasoning_tok_opt, finish_reason_opt) =
        exec_once(client, &body, id_str, key, logger).await?;

    // If we got non-empty content, return immediately
    if !content.trim().is_empty() {
        let mut final_text = content;
        if max_output_tokens > 0 {
            final_text = truncate_to_tokens(&final_text, max_output_tokens);
        }
        return Ok((final_text, p_tok, c_tok));
    }

    // Detect “reasoning starvation”
    let starved = finish_reason_opt.as_deref() == Some("length")
        && content.trim().is_empty()
        // if the API splits completion into reasoning vs visible, "all went to reasoning"
        && reasoning_tok_opt.unwrap_or(0) >= c_tok;

    //if starved && fallback_on_empty {
    if starved {
        logger.log(&format!(
            "ID {} key {}: Starved. Retrying with reasoning.effort=minimal and +64 token headroom.",
            id_str, key
        ));
        let fb_cap_param = cap_param_for_model(model);
        let mut fb_body = build_body(model, temperature, fb_cap_param);
        if let Some(obj) = fb_body.as_object_mut() {
            obj.insert("reasoning".into(), json!({"effort":"minimal"}));
            if max_output_tokens > 0 { obj.insert(fb_cap_param.into(), json!(max_output_tokens + 64)); }
        }
        let (_j, fb_content, fb_p, fb_c, _r, _fr) = exec_once(client, &fb_body, id_str, key, logger).await?;
        if !fb_content.trim().is_empty() {
            let mut final_text = fb_content;
            if max_output_tokens > 0 { final_text = truncate_to_tokens(&final_text, max_output_tokens); }
            return Ok((final_text, fb_p, fb_c));
        }
    }
    if starved && fallback_on_empty {
        // Fail open to non-reasoning chat model (one attempt)
        logger.log(&format!(
            "ID {} key {}: Output starved by reasoning ({} reasoning tokens). Falling back to {}.",
            id_str, key, reasoning_tok_opt.unwrap_or(0), fallback_model
        ));
        let fb_cap_param = cap_param_for_model(fallback_model);
        let fb_body = build_body(fallback_model, temperature, fb_cap_param);
        let (_fb_json, fb_content, fb_p_tok, fb_c_tok, _fb_reasoning, _fb_finish) =
            exec_once(client, &fb_body, id_str, key, logger).await?;

        if !fb_content.trim().is_empty() {
            let mut final_text = fb_content;
            if max_output_tokens > 0 {
                final_text = truncate_to_tokens(&final_text, max_output_tokens);
            }
            return Ok((final_text, fb_p_tok, fb_c_tok));
        } else {
            return Err(anyhow!("Empty content after fallback to {}", fallback_model));
        }
    }

    // No fallback, or not starved: treat as error so it’s visible in the logs/output
    Err(anyhow!("Empty content in successful response"))
}
