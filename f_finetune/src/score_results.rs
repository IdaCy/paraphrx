/*
Use: cargo run --release --bin results_assess_api_limited -- \
  --model gemini-2.0-flash \
  --api-call-max 500 \
  f_finetune/data/alpaca_gemma-2-2b-it.json \
  c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/instruct_merged/buckets1.json \
  out/buckets1_scored.json
*/

use anyhow::{anyhow, Context, Result};
use chrono::Local;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value};
use std::{
    collections::{HashMap, HashSet},
    fs, io::BufWriter,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};
use tokio::time::sleep;
use std::io::Write;

// little logger
struct Logger {
    writer: BufWriter<fs::File>,
}
impl Logger {
    fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(p)?;
        Ok(Self { writer: BufWriter::new(file) })
    }
    fn log(&mut self, msg: &str) {
        let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(self.writer, "[{ts}] {msg}");
        let _ = self.writer.flush();
    }
}
// accept either 123  or  "123"  for prompt_count
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
                .map_err(|_| E::custom(format!("invalid prompt_count {}", v)))
        }
    }
    de.deserialize_any(Visitor)
}

// data structs
#[derive(Debug, Deserialize, Serialize, Clone)]
struct Record {
    #[serde(alias = "prompt_count", deserialize_with = "de_prompt_count")]
    prompt_count: u32,

    #[serde(alias = "instruction", alias = "instruction_original", default)]
    instruction_original: String,

    #[serde(default)]
    output: Option<String>,

    // Everything else (all paraphrase keys, etc.) lands here
    #[serde(flatten)]
    extra: JsonMap<String, Value>,
}

// CLI
#[derive(Parser, Debug)]
#[command(version, author, about = "Assess paraphrase answers with Gemini; hard-limit total API calls")] 
struct Cli {
    // Original instructions & paraphrases
    instructions: PathBuf,
    // Inference answers JSON produced by the fine-tuned model
    answers: PathBuf,
    output: PathBuf,

    // Gemini model name (e.g. "gemini-2.5-flash-preview-05-20")
    #[arg(long, default_value = "gemini-2.0-flash")]
    model: String,

    // Max attempts per prompt when an evaluation call fails
    #[arg(long, default_value_t = 5)]
    max_attempts: u8,

    // Milliseconds to sleep after a successful first-try call (helps avoid 429)
    #[arg(long = "delay-ms", default_value_t = 200)]
    delay_ms: u64,

    // Google API key; overrides $GOOGLE_API_KEY
    #[arg(long = "api-key", value_name = "KEY")]
    api_key: Option<String>,

    // Global hard-cap on total Gemini calls across the entire run - stops+saves!
    #[arg(long = "api-call-max", default_value_t = 10_000)]
    api_call_max: u32,
}

// helpers
fn schema_for_keys(keys: &[String]) -> Value {
    let mut props = JsonMap::new();
    for k in keys {
        props.insert(
            k.clone(),
            json!({"type":"array","items":{"type":"integer"},"minItems":10,"maxItems":10}),
        );
    }
    json!({"type":"object","properties":props,"required":keys})
}

// fault-tolerant JSON loader
fn read_records(path: &Path, logger: &mut Logger) -> HashMap<String, Record> {
    match fs::read_to_string(path)
        .and_then(|s| serde_json::from_str::<Vec<Record>>(&s).map_err(Into::into))
    {
        Ok(vec) => vec
            .into_iter()
            .map(|r| (r.prompt_count.to_string(), r))   // key stays String
            .collect(),
        Err(e) => {
            logger.log(&format!("[fatal-but-skipped] could not parse {}: {e}", path.display()));
            HashMap::new() // empty -> main loop skips gracefully
        }
    }
}

const ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";

// main
#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // logging
    let log_dir = Path::new("logs");
    fs::create_dir_all(log_dir)?;
    let ts = Local::now().format("%Y%m%d-%H%M%S");
    let stem = cli
        .output
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    let log_path = log_dir.join(format!("{stem}_{ts}.logs"));
    let mut logger = Logger::new(&log_path)?;
    logger.log(&format!(
        "run started -> model={}, api_call_max={}, log={}",
        cli.model,
        cli.api_call_max,
        log_path.display()
    ));

    // I/O
    logger.log("reading json files");
    let instr_map = read_records(&cli.instructions, &mut logger);
    let ans_map = read_records(&cli.answers, &mut logger);

    // check inputs
    if instr_map.is_empty() {
        return Err(anyhow!("instructions JSON is empty or unreadable"));
    }
    if ans_map.is_empty() {
        return Err(anyhow!("answers JSON is empty or unreadable"));
    }

    // collect & numerically sort ids for deterministic order
    let mut instr_sorted: Vec<(&String, &Record)> = instr_map.iter().collect();
    instr_sorted.sort_by_key(|(_, r)| r.prompt_count);
    let all_ids: Vec<String> = instr_sorted.iter().map(|(id, _)| (*id).clone()).collect();

    // HTTP client
    let api_key = cli
        .api_key
        .clone()
        .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .context("provide --api-key or set GOOGLE_API_KEY")?;
    let client = build_client()?;

    // progress bar
    let bar = ProgressBar::new(instr_sorted.len() as u64);
    bar.set_style(ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
    )?);

    // run bookkeeping
    let mut results = Vec::new();
    let mut issues = Vec::new();
    let mut processed_ids: HashSet<String> = HashSet::new();
    let mut api_calls_used: u32 = 0;
    let mut early_stop = false;

    for (id, inst) in instr_sorted {
        // pre-stop gate
        if api_calls_used >= cli.api_call_max {
            logger.log(&format!(
                "api_call_max={} reached ({} calls used) -> early stop",
                cli.api_call_max, api_calls_used
            ));
            early_stop = true;
            break;
        }

        logger.log(&format!("▶ id {id}"));
        let (calls_used, success) = match process_single(
            id,
            inst,
            &ans_map,
            &client,
            &api_key,
            &cli.model,
            cli.max_attempts,
            &mut logger,
            &mut results,
            &mut issues,
        )
        .await
        {
            Ok(pair) => pair,
            Err(e) => {
                logger.log(&format!("[error] id {id}: {e}"));
                issues.push(format!("id {id}: {e}"));
                (0, false) // no calls, not successful
            }
        };

        api_calls_used += calls_used as u32;
        if success {
            processed_ids.insert(id.clone());
        }

        bar.inc(1);

        // gentle rate-limit pause (first-try success only)
        if cli.delay_ms > 0 && calls_used == 1 {
            sleep(Duration::from_millis(cli.delay_ms)).await;
        }
    }
    bar.finish_with_message("done / saving");

    // outputs
    fs::write(&cli.output, serde_json::to_string_pretty(&results)?)?;
    logger.log("results written");

    if !issues.is_empty() {
        let issues_path = cli.output.with_extension("issues.json");
        fs::write(&issues_path, serde_json::to_string_pretty(&issues)?)?;
        logger.log(&format!(
            "wrote {} issues to {}",
            issues.len(),
            issues_path.display()
        ));
    }

    // unprocessed IDs when early stop
    if early_stop {
        let mut missing: Vec<String> = all_ids
            .into_iter()
            .filter(|id| !processed_ids.contains(id))
            .map(|id| format!("id {id}: unprocessed"))
            .collect();
        missing.sort();
        let miss_path = cli.output.with_extension("unprocessed.json");
        fs::write(&miss_path, serde_json::to_string_pretty(&missing)?)?;
        logger.log(&format!("wrote {} unprocessed IDs to {}", missing.len(), miss_path.display()));
        println!(
            "early stop ({} API calls used) - {} IDs unprocessed - log {}",
            api_calls_used,
            missing.len(),
            log_path.display()
        );
    } else if issues.is_empty() {
        println!("done - log {}", log_path.display());
    } else {
        println!(
            "done with {} issues - log {}",
            issues.len(),
            log_path.display()
        );
    }

    Ok(())
}

// processing one prompt_count
async fn process_single(
    id: &str,
    inst: &Record,
    ans_map: &HashMap<String, Record>,
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    max_attempts: u8,
    logger: &mut Logger,
    results: &mut Vec<Value>,
    issues: &mut Vec<String>,
) -> Result<(u8 /*calls*/, bool /*success*/)> {
    let ans = match ans_map.get(id) {
        Some(a) => a,
        None => {
            issues.push(format!("answers missing id {id}"));
            return Ok((0, false));
        }
    };

    // collect all instruct_* keys across instruction & answer JSONs
    let mut keys: Vec<String> = vec!["instruction_original".to_string()];
    keys.extend(
        inst.extra
            .keys()
            .chain(ans.extra.keys())
            .filter(|k| k.starts_with("instruct_"))
            .cloned(),
    );
    keys.sort();
    keys.dedup();

    // build evaluation section (instruction + answer for each paraphrase key)
    let mut section = String::new();
    for key in &keys {
        let instr_text = inst
            .extra
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or(&inst.instruction_original);
        let ans_text = ans
            .extra
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or(&ans.instruction_original);
        section.push_str(&format!(
            "### {key}\n[Instruction]\n{instr_text}\n\n[Answer]\n{ans_text}\n\n"
        ));
    }
    if section.len() > 95_000 {
        issues.push(format!("id {id}: prompt too large"));
        return Ok((0, false));
    }

    //let schema = schema_for_keys(&keys);
    let prompt = build_eval_prompt(&section);

    // attempt loop
    let mut eval_json = JsonMap::new();
    for attempt in 1..=max_attempts {
        logger.log(&format!("[call] id {id} attempt {attempt}/{max_attempts}"));
        //match query_gemini(client, api_key, model, schema.clone(), prompt.clone()).await {
        match query_gemini(client, api_key, model, Value::Null, prompt.clone()).await {
        Ok(obj) => {
                logger.log(&format!("[ok]   id {id} attempt {attempt}/{max_attempts}"));
                eval_json = obj;

                // store result object
                let mut res = JsonMap::new();
                // try to preserve numeric if parseable
                res.insert("prompt_count".to_string(), json!(inst.prompt_count));
                for key in &keys {
                    if let Some(v) = eval_json.get(key) {
                        res.insert(key.clone(), v.clone());
                    } else {
                        issues.push(format!("id {id}: missing eval key {key}"));
                    }
                }
                results.push(Value::Object(res));
                logger.log(&format!("[done] id {id} fully processed"));
                return Ok((attempt, true)); // attempt calls used, success
            }
            Err(e) if attempt < max_attempts => {
                logger.log(&format!("[warn] id {id} attempt {attempt}/{max_attempts}: {e}"));
                // exponential back-off jitter
                let wait_ms = 500u64 * 2u64.pow(attempt as u32)
                    + (SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .subsec_millis() as u64)
                        % 300;
                sleep(Duration::from_millis(wait_ms)).await;
            }
            Err(e) => {
                issues.push(format!("id {id}: {e}"));
                return Ok((attempt, false)); // calls used = attempt, failed
            }
        }
    }

    issues.push(format!("id {id}: all attempts failed"));
    Ok((max_attempts, false))
}

// HTTP helpers
fn build_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Ok(reqwest::Client::builder().default_headers(headers).build()?)
}

fn build_eval_prompt(section: &str) -> String {
    format!(r#"You are an expert evaluator.

For every answer below, assess it against **ten metrics**. Each metric must be scored on a 0-10 integer scale (higher is better).

Metrics (use **exact** order):
1. Task Fulfilment / Relevance - Does it respond to every part of the prompt? Did it wander off-topic or over-answer?
2. Usefulness & Actionability - Does it translate abstract ideas into concrete advice, examples, or next steps?
3. Factual Accuracy & Verifiabiliy - Are the statements factually correct (no hallucinations)? If it cites sources or internal steps, do those match the final claims?
4. Efficiency / Depth & Completeness - Does it avoid unnecessary verbosity or excessive brevity? Does it cover the key angles, edge-cases, and typical follow-ups? Could the user act on it without having to ask “what about X?”
5. Reasoning Quality / Transparency - Are the steps implicitly or explicitly sound? If uncertain, does it flag that uncertainty instead of bluffing?
6. Tone & Likeability - Is the style friendly and respectful, matching the user’s vibe? Would you enjoy a longer conversation in this voice?
7. Adaptation to Context - Does it use any relevant info the user has shared (location, preferences, prior messages) appropriately?
8. Safety & Bias Avoidance - Does it steer clear of harmful or disallowed content? Does it acknowledge and mitigate possible bias?
9. Structure & Formatting & UX Extras - Is the writing logically ordered and easy to skim? Are lists, code blocks, tables, or rich widgets used when, but only when they genuinely improve readability or utility?
10. Creativity - Does the answer make clever, non-obvious connections that you wouldn’t get from a quick Google search? Or, does it remix ideas, metaphors, or examples in a fresh way rather than serving boilerplate?

Return **only** JSON conforming to this Pydantic model (nothing else, no code fences):

```python
from typing import Dict, List
from pydantic import BaseModel, conlist

class EvalResult(BaseModel):
    __root__: Dict[str, conlist(int, min_items=10, max_items=10)]
```

Begin data to evaluate:

{section}
"#)
}

async fn query_gemini(
    client: &reqwest::Client,
    key: &str,
    model: &str,
    schema: Value,
    prompt: String,
) -> Result<JsonMap<String, Value>> {
    let url = format!("{ENDPOINT}/models/{}:generateContent?key={}", model, key);
    let body = json!({
        "contents": [{
            "role": "user",
            "parts": [{ "text": prompt }]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    });
    let resp = client.post(&url).json(&body).send().await?;
    if !resp.status().is_success() {
        return Err(anyhow!("{} — {}", resp.status(), resp.text().await?));
    }
    let resp_json: Value = resp.json().await?;
    let json_text = resp_json["candidates"][0]["content"]["parts"][0]["text"].as_str()
        .ok_or_else(|| anyhow!("unexpected response structure"))?;
    Ok(serde_json::from_str(json_text.trim())?)
}
