/*
cargo results_assess \
  --model gemini-2.0-flash \
  a_data/alpaca/merge_instructs/all.json \
  c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/instruct_merged/all.json \
  c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/instruct_merged/all_results.json
*/

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value};
use chrono::Local;
use std::{
    collections::HashMap,
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};
use tokio::time::sleep;

// tiny rolling logger
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

// data structs
#[derive(Debug, Deserialize, Serialize, Clone)]
struct Record {
    prompt_count: u32,
    #[serde(alias = "instruction", alias = "instruction_original")]
    instruction_original: String,
    #[serde(default)]
    output: Option<String>,
    #[serde(flatten)]
    extra: JsonMap<String, Value>,
}

// CLI
#[derive(Parser, Debug)]
#[command(version, author, about = "Assess paraphrase answers with Gemini")]
struct Cli {
    instructions: PathBuf,
    answers: PathBuf,
    output: PathBuf,

    // Gemini model name (e.g. gemini-2.5-flash-preview-05-20)
    #[arg(long, default_value = "gemini-2.0-flash")]
    model: String,

    #[arg(long, default_value_t = 5)]
    max_attempts: u8,

    // Milliseconds to wait after every successful request (avoid 429s)
    #[arg(long = "delay-ms", default_value_t = 200)]
    delay_ms: u64,

    // Google API key (overrides $GOOGLE_API_KEY)
    #[arg(long = "api-key", value_name = "KEY")]
    api_key: Option<String>,
}

fn schema_for_keys(keys: &[String]) -> Value {
    let mut props = JsonMap::new();
    for k in keys {
        props.insert(k.clone(), json!({"type":"array","items":{"type":"integer"},"minItems":10,"maxItems":10}));
    }
    json!({"type":"object","properties":props,"required":keys})
}

// fault-tolerant JSON loader
fn read_records(path: &Path, logger: &mut Logger) -> HashMap<String, Record> {
    match fs::read_to_string(path)
        .and_then(|s| serde_json::from_str::<Vec<Record>>(&s).map_err(Into::into))
    {
        Ok(vec) => vec.into_iter()
                      .map(|r| (r.prompt_count.to_string(), r))
                      .collect(),
        Err(e) => {
            logger.log(&format!(
                "[fatal-but-skipped] could not parse {}: {e}", path.display()
            ));
            HashMap::new()                      // empty -> loop simply skips
        }
    }
}

const ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // global log directory
    let log_dir = Path::new("logs");
    fs::create_dir_all(log_dir)?;
    let ts = Local::now().format("%Y%m%d-%H%M%S");

    let stem = cli
        .output
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();

    // logs/<stem>_<timestamp>.logs
    let log_path = log_dir.join(format!("{stem}_{ts}.logs"));

    let mut logger = Logger::new(&log_path)?;
    logger.log(&format!("run started -> model={} log={}", cli.model, log_path.display()));

    // I/O
    logger.log("reading json files");
    let instr_map = read_records(&cli.instructions, &mut logger);
    let ans_map   = read_records(&cli.answers,     &mut logger);

    let api_key = cli
        .api_key
        .clone()
        .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .context("provide --api-key or set GOOGLE_API_KEY")?;
    let client  = build_client()?;

    // sort so we run strictly in prompt_count order
    let mut instr_sorted: Vec<(&String, &Record)> = instr_map.iter().collect();
    instr_sorted.sort_by_key(|(_, r)| r.prompt_count);

    let bar = ProgressBar::new(instr_sorted.len() as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )?,
    );

    let mut results = Vec::new();
    let mut issues  = Vec::new();

    for (id, inst) in instr_sorted {
        logger.log(&format!("▶ id {id}"));
        let attempts = match process_single(
            id, inst, &ans_map, &client, &api_key, &cli.model,
            cli.max_attempts, &mut logger, &mut results, &mut issues,
        ).await {
            Ok(n) => n,             // number of tries actually used
            Err(e) => {
                logger.log(&format!("[error] id {id}: {e}"));
                issues.push(format!("id {id}: {e}"));
                cli.max_attempts     // treat as 'slow' so we skip sleep
            }
        };
        bar.inc(1);
        // Global rate-limit pause (configurable via --delay-ms)
        // Pause only if it flew through on the first go
        if cli.delay_ms > 0 && attempts == 1 {
            sleep(Duration::from_millis(cli.delay_ms)).await;
        }
    }
    bar.finish_with_message("done");

    fs::write(&cli.output, serde_json::to_string_pretty(&results)?)?;
    logger.log("results written");

    if !issues.is_empty() {
        let issues_path = cli.output.with_extension("issues.json");
        fs::write(&issues_path, serde_json::to_string_pretty(&issues)?)?;
        logger.log(&format!(
            "wrote {} issues to {}", issues.len(), issues_path.display()
        ));
    }

    if issues.is_empty() {
        println!("done - log {}", log_path.display());
    } else {
        println!("done with {} issues - log {}", issues.len(), log_path.display());
    }
    Ok(())
}

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
) -> Result<u8> {
    let ans = match ans_map.get(id) {
        Some(a) => a,
        None => {
            issues.push(format!("answers missing id {id}"));
            return Ok(max_attempts);
        }
    };
    let mut keys = vec!["instruction_original".to_string()];
    keys.extend(
        inst
            .extra
            .keys()
            .chain(ans.extra.keys())
            .filter(|k| k.starts_with("instruct_"))
            .cloned(),
    );
    keys.sort();
    keys.dedup();

    let mut section = String::new();
    for key in &keys {
        let instr = inst
            .extra
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or(&inst.instruction_original);
        let ans_txt = ans
            .extra
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or(&ans.instruction_original);
        section.push_str(&format!(
            "### {key}\n[Instruction]\n{instr}\n\n[Answer]\n{ans_txt}\n\n"
        ));
    }
    if section.len() > 95_000 {
        issues.push(format!("id {id}: prompt too large"));
        return Ok(max_attempts);
    }

    let schema = schema_for_keys(&keys);
    let prompt = build_eval_prompt(&section);
    let mut success = false;
    let mut eval_json = JsonMap::new();
    let mut attempts_used = max_attempts;
    for attempt in 1..=max_attempts {
        logger.log(&format!(
            "[call] id {id} attempt {attempt}/{max_attempts}"
        ));

        match query_gemini(client, api_key, model, schema.clone(), prompt.clone()).await {
            Ok(obj) => {
                logger.log(&format!(
                    "[ok]   id {id} attempt {attempt}/{max_attempts}"
                ));

                eval_json = obj;
                success = true;
                attempts_used = attempt;
                break;
            }
            Err(e) if attempt < max_attempts => {
                let wait = 500u64 * 2u64.pow(attempt as u32)
                    + (SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .subsec_millis() as u64) % 300;

                logger.log(&format!(
                    "[warn] id {id} attempt {attempt}/{max_attempts}: {e}"
                ));
                sleep(Duration::from_millis(wait)).await;
            }
            Err(e) => {
                issues.push(format!("id {id}: {e}"));
                return Ok(max_attempts);
            }
        }
    }
    if !success {
        issues.push(format!("id {id}: all attempts failed"));
        return Ok(max_attempts);
    }

    let mut res_obj = JsonMap::new();
    res_obj.insert(
        "prompt_count".to_string(),
        serde_json::to_value(inst.prompt_count)?,
    );
    for key in &keys {
        if let Some(v) = eval_json.get(key) {
            res_obj.insert(key.clone(), v.clone());
        } else {
            issues.push(format!("id {id}: missing eval key {key}"));
        }
    }
    results.push(Value::Object(res_obj));
    logger.log(&format!("[done] id {id} fully processed"));
    Ok(attempts_used)
}

fn build_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Ok(reqwest::Client::builder().default_headers(headers).build()?)
}

fn build_eval_prompt(section: &str) -> String {format!(r#"You are an expert evaluator.

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
"#)}

async fn query_gemini(
    client:&reqwest::Client,
    key:&str,
    model:&str,
    schema:Value,
    prompt:String,
)->Result<JsonMap<String,Value>>{
    //let url=format!("{ENDPOINT}/models/{}/generateContent?key={}",model,key);
    let url = format!("{ENDPOINT}/models/{}:generateContent?key={}", model, key);
    let body=json!({"contents":[{"role":"user","parts":[{"text":prompt}]}],"generationConfig":{"responseMimeType":"application/json","responseSchema":schema}});
    let resp=client.post(&url).json(&body).send().await?;
    if !resp.status().is_success(){return Err(anyhow!("{} — {}",resp.status(),resp.text().await?));}
    let resp_json:Value=resp.json().await?;
    let json_text=resp_json["candidates"][0]["content"]["parts"][0]["text"].as_str().ok_or_else(||anyhow!("unexpected response structure"))?;
    Ok(serde_json::from_str(json_text.trim())?)
}
