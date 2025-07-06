/*
cargo phrx_equivalence_score \
  --model gemini-2.0-flash \
  --api-key $GOOGLE_API_KEY \
  --delay-ms 200 \
  --api-call-maximum 250 \
  data/prompts_with_paraphrases.json \
  output/paraphrase_scores.json
*/

use anyhow::{anyhow, Context, Result};
use chrono::Local;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value};
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

// data model
#[derive(Debug, Deserialize, Serialize, Clone)]
struct Record {
    prompt_count: u32,
    #[serde(alias = "instruction", alias = "instruction_original")]
    instruction_original: String,

    // everything else (all instruct_* variations, inputs, etc.)
    #[serde(flatten)]
    extra: JsonMap<String, Value>,
}

// CLI
#[derive(Parser, Debug)]
#[command(version, author, about = "Assess paraphrase similarity (content-level)")]
struct Cli {
    // JSON file containing the prompts + paraphrases (see README)
    prompts: PathBuf,

    // Where to write the scored JSON file
    output: PathBuf,

    // Gemini model name
    #[arg(long, default_value = "gemini-2.0-flash")]
    model: String,

    // Maximum attempts per individual API call (for transient failures)
    #[arg(long, default_value_t = 5)]
    max_attempts: u8,

    // Milliseconds to wait after every *successful* request
    #[arg(long = "delay-ms", default_value_t = 200)]
    delay_ms: u64,

    // Hard ceiling of total API calls in a single run (progress is saved)
    #[arg(long = "api-call-maximum", default_value_t = 250)]
    api_call_maximum: usize,

    // Google API key (overrides $GOOGLE_API_KEY env-var)
    #[arg(long = "api-key", value_name = "KEY")]
    api_key: Option<String>,
}

// JSON Schema builder
//fn schema_for_keys(_keys: &[String]) -> Value {
//    json!({ "type": "object" })
//}

// helper: read & parse
fn read_records<P: AsRef<Path>>(path: P, logger: &mut Logger) -> HashMap<u32, Record> {
    match fs::read_to_string(&path).and_then(|s| serde_json::from_str::<Vec<Record>>(&s).map_err(Into::into)) {
        Ok(vec) => vec.into_iter().map(|r| (r.prompt_count, r)).collect(),
        Err(e) => {
            logger.log(&format!("[fatal] could not parse {}: {e}", path.as_ref().display()));
            HashMap::new()
        }
    }
}

// BUILD PROMPT
fn build_eval_prompt(section: &str) -> String {
    format!(
        r#"You are an expert linguistic evaluator specialised in semantic equivalence.

For *each* paraphrase below (keys starting with `instruct_`), compare its **requested content** to the `instruction_original`. Do **not** care about wording similarity - only whether they ask for the *same thing*. Score every paraphrase on this integer scale:

10 - valid paraphrase; identical information request, merely reworded
9  - fine paraphrase; very small shift in requested details (slight extra/omission)
8  - still a paraphrase; noticeable added/omitted element but core request intact
7  - partial; missing or adding substantive element → clearly different answer needed
6  - different content dominates; only loosely related to original
5  - overlaps in topic but mainly asks something else
4  - faint topical intersection only
3  - original topic barely present
2  - almost totally unrelated
1  - completely unrelated
0  - unrelated and essentially not a coherent request

Return **ONLY** JSON that conforms *exactly* to this Pydantic model (no extra keys, no code-fences):

```python
from typing import Dict
from pydantic import BaseModel, conint

class ParaphraseScore(BaseModel):
    __root__: Dict[str, conint(ge=0, le=10)]
```

Begin items:

{section}
"#
    )
}

// Gemini call helper
const ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";
async fn query_gemini(
    client: &reqwest::Client,
    key: &str,
    model: &str,
    //schema: Value,
    prompt: String,
) -> Result<JsonMap<String, Value>> {
    let url = format!("{ENDPOINT}/models/{model}:generateContent?key={key}");
    let body = json!({
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            //"responseSchema": schema,
            "temperature": 0.0,
        }
    });

    let resp = client.post(&url).json(&body).send().await?;
    if !resp.status().is_success() {
        return Err(anyhow!("{} — {}", resp.status(), resp.text().await?));
    }
    let resp_json: Value = resp.json().await?;
    let json_text = resp_json["candidates"][0]["content"]["parts"][0]["text"].as_str()
        .ok_or_else(|| anyhow!("unexpected response structure"))?;
    let cleaned = json_text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    Ok(serde_json::from_str(cleaned)?)
}

// main async 
#[tokio::main(flavor = "multi_thread")] // allows many concurrent sleeps
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // logging setup
    let log_dir = Path::new("logs");
    fs::create_dir_all(log_dir)?;
    let ts = Local::now().format("%Y%m%d-%H%M%S");
    let stem = cli.output.file_stem().unwrap_or_default().to_string_lossy();
    let log_path = log_dir.join(format!("{stem}_{ts}.logs"));
    let mut logger = Logger::new(&log_path)?;
    logger.log(&format!("run started → model={} log={}", cli.model, log_path.display()));

    // data ingest
    logger.log("reading prompts JSON");
    let records = read_records(&cli.prompts, &mut logger);
    if records.is_empty() {
        return Err(anyhow!("no valid records loaded"));
    }

    // sort strictly by prompt_count asc
    let mut ordered: Vec<&Record> = records.values().collect();
    ordered.sort_by_key(|r| r.prompt_count);

    // api key + client
    let api_key = cli
        .api_key
        .clone()
        .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .context("provide --api-key or set GOOGLE_API_KEY")?;
    let client = build_client()?;

    // progress tracking
    let bar = ProgressBar::new(ordered.len() as u64);
    bar.set_style(
        ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?,
    );

    let mut results: Vec<Value> = Vec::new();
    let mut issues: Vec<String> = Vec::new();
    let mut calls_made: usize = 0;

    for record in ordered {
        let id = record.prompt_count;

        if calls_made >= cli.api_call_maximum {
            issues.push(format!("id {id}: unprocessed"));
            continue;
        }

        // gather instruct_* keys
        let mut keys: Vec<String> = record
            .extra
            .keys()
            .filter(|k| k.starts_with("instruct_"))
            .cloned()
            .collect();
        keys.sort();

        if keys.is_empty() {
            issues.push(format!("id {id}: no instruct_* fields found"));
            continue;
        }

        // build section text for prompt
        let mut section = String::new();
        for key in &keys {
            let paraphrase = record.extra[key].as_str().unwrap_or("");
            section.push_str(&format!(
                "### {key}\n[Instruction original]\n{}\n\n[Paraphrase]\n{}\n\n",
                record.instruction_original, paraphrase
            ));
        }

        // check token / size limit (Gemini 100k char safe guard)
        //const MAX_CHARS: usize = 40_000;   // ~10 k tokens ⇒ well under 12 288
        //if section.len() > MAX_CHARS {
        //    issues.push(format!("id {id}: prompt too large, skipping"));
        //    continue;
        //}

        // schema & prompt
        //let schema = schema_for_keys(&keys);
        let prompt = build_eval_prompt(&section);

        // attempt loop
        let mut success = false;
        let mut eval_json = JsonMap::new();
        let mut attempts_used = 0;
        for attempt in 1..=cli.max_attempts {
            logger.log(&format!("[call] id {id} attempt {attempt}/{}/{}", attempt, cli.max_attempts));

            match query_gemini(&client, &api_key, &cli.model, prompt.clone()).await {
                Ok(obj) => {
                    logger.log(&format!("[ok]   id {id} attempt {attempt}"));
                    eval_json = obj;
                    success = true;
                    attempts_used = attempt;
                    break;
                }
                Err(e) if attempt < cli.max_attempts => {
                    let wait = 500u64 * 2u64.pow(attempt as u32)
                        + (SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().subsec_millis() as u64) % 300;
                    logger.log(&format!("[warn] id {id} attempt {attempt}: {e}"));
                    sleep(Duration::from_millis(wait)).await;
                }
                Err(e) => {
                    issues.push(format!("id {id}: {e}"));
                    break;
                }
            }
        }

        if !success {
            bar.inc(1);
            continue; // go next id, keep issues note
        }

        // build result object
        let mut obj = JsonMap::new();
        obj.insert("prompt_count".to_string(), json!(id));
        // include perfect 10 for original explicitly
        obj.insert("instruction_original".to_string(), json!(10));
        for key in &keys {
            if let Some(v) = eval_json.get(key) {
                obj.insert(key.clone(), v.clone());
            } else {
                issues.push(format!("id {id}: missing score for {key}"));
            }
        }
        results.push(Value::Object(obj));
        calls_made += 1;
        bar.inc(1);

        // global rate limit sleep
        if cli.delay_ms > 0 && attempts_used == 1 {
            sleep(Duration::from_millis(cli.delay_ms)).await;
        }
    }
    bar.finish_with_message("done");

    // write outputs
    fs::create_dir_all(cli.output.parent().unwrap_or(Path::new(".")))?;
    fs::write(&cli.output, serde_json::to_string_pretty(&results)?)?;
    logger.log("results written");

    // write issues (includes unprocessed list)
    if !issues.is_empty() {
        let issues_path = cli.output.with_extension("issues.json");
        fs::write(&issues_path, serde_json::to_string_pretty(&issues)?)?;
        logger.log(&format!("wrote {} issues to {}", issues.len(), issues_path.display()));
    }

    println!(
        "done - {} processed, {} issues - log {}",
        results.len(),
        issues.len(),
        log_path.display()
    );

    Ok(())
}

// reqwest client builder
fn build_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Ok(reqwest::Client::builder().default_headers(headers).build()?)
}
