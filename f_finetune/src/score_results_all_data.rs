/*
cargo score_results_all_data \
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
use once_cell::sync::Lazy;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value};
use std::{
    collections::{HashMap, HashSet},
    fs, io::BufWriter,
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::time::sleep;
use std::io::Write;

const PROMPT_PREAMBLE_TOKENS: usize = 500;
const DEBUG_IDS: &[u32] = &[1, 42, 311];

// Constants
static MODEL_LIMITS: Lazy<HashMap<&'static str, usize>> = Lazy::new(|| {
    HashMap::from([
        ("gemini-2.5-flash-preview-05-20", 1_048_576),
        ("gemini-2.5-flash-lite-preview-06-17", 1_000_000),
        ("gemini-2.5-flash", 1_048_576),
        ("gemini-2.5-pro", 1_048_576),
        ("gemini-2.0-flash", 1_048_576),
    ])
});

const ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";

// Misc helpers
fn estimate_tokens(text: &str) -> usize {
    // very rough: 0.75 * words ≈ tokens   (≈ bytes / 4)
    ((text.split_whitespace().count() as f32) * 0.75).ceil() as usize
}

// Logger
struct Logger {
    writer: BufWriter<fs::File>,
}
impl Logger {
    fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
    fn log(&mut self, msg: &str) {
        let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(self.writer, "[{ts}] {msg}");
        let _ = self.writer.flush();
    }
}

// Data model
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

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Record {
    #[serde(alias = "prompt_count", deserialize_with = "de_prompt_count")]
    prompt_count: u32,

    #[serde(alias = "instruction", alias = "instruction_original", default)]
    instruction_original: String,

    #[serde(default)]
    output: Option<String>,

    #[serde(flatten)]
    extra: JsonMap<String, Value>,
}

// CLI
#[derive(Parser, Debug)]
#[command(version, about = "Assess paraphrase answers with Gemini (resume-able, token-aware)")]
struct Cli {
    // one or more instruction JSONs (e.g. alpaca.json gsm8k.json mmlu.json)
    #[arg(long = "instructions", required = true)]
    instructions: Vec<PathBuf>,
    // matching labels, same length as --instructions -> --datasets alpaca gsm8k mmlu
    #[arg(long = "datasets", required = true)]
    datasets: Vec<String>,

    answers: PathBuf,
    output: PathBuf,

    #[arg(long, default_value = "gemini-2.0-flash")]
    model: String,

    #[arg(long = "log-name", default_value = "SCORING")]
    log_name: String,

    #[arg(long, default_value_t = 5)]
    max_attempts: u8,

    #[arg(long = "delay-ms", default_value_t = 200)]
    delay_ms: u64,

    #[arg(long = "api-key")]
    api_key: Option<String>,

    #[arg(long = "api-call-max", default_value_t = 10_000)]
    api_call_max: u32,

    // keep at least this many tokens below the model context limit
    #[arg(long, default_value_t = 2048)]
    margin: usize,

    // emergency upper bound on instruct_* per chunk
    #[arg(long = "chunk-max", default_value_t = 163)]
    chunk_max: usize,
}

// JSON helpers
fn read_records(path: &Path, logger: &mut Logger) -> HashMap<String, Record> {
    match fs::read_to_string(path)
        .and_then(|s| serde_json::from_str::<Vec<Value>>(&s).map_err(Into::into))
    {
        Ok(items) => items
            .into_iter()
            .map(|raw| {
                // turn the raw JSON into our Record
                let mut rec: Record = serde_json::from_value(raw.clone())?;

                // pull paraphrases into `extra`
                if let Some(pars) = raw.get("paraphrases").and_then(Value::as_array) {
                    for p in pars {
                        if let (Some(t), Some(txt)) = (
                            p.get("instruct_type").and_then(Value::as_str),
                            p.get("paraphrase").and_then(Value::as_str),
                        ) {
                            rec.extra
                                .insert(t.to_string(), Value::String(txt.to_string()));
                        }
                    }
                }
                Ok((rec.prompt_count.to_string(), rec))
            })
            .collect::<Result<_, anyhow::Error>>() // propagate errors
            .unwrap_or_else(|e| {
                logger.log(&format!("[fatal] {e}"));
                HashMap::new()
            }),
        Err(e) => {
            logger.log(&format!("[fatal-but-skipped] cannot parse {}: {e}", path.display()));
            HashMap::new()
        }
    }
}

// Read inference‐output JSON (with fields uid, dataset, prompt_count, plus
// all the "instruct_*": answer keys) into a HashMap keyed by uid
fn read_answers(path: &Path, logger: &mut Logger) -> HashMap<String, AnsRecord> {
    let raw: Vec<Value> = match fs::read_to_string(path) {
        Ok(s) => match serde_json::from_str(&s) {
            Ok(v) => v,
            Err(e) => {
                logger.log(&format!("[fatal] parsing answers JSON: {e}"));
                return HashMap::new();
            }
        },
        Err(e) => {
            logger.log(&format!("[fatal] reading answers JSON: {e}"));
            return HashMap::new();
        }
    };

    let mut map = HashMap::new();
    for v in raw {
        // ensure JSON object
        let obj = if let Some(o) = v.as_object() {
            o
        } else {
            logger.log(&format!("skipping non-object entry in answers JSON"));
            continue;
        };
        // extract and validate uid and dataset
        let uid = if let Some(s) = obj.get("uid").and_then(Value::as_str).filter(|s| !s.is_empty()) {
            s.to_string()
        } else {
            logger.log("missing or empty uid — skipping record");
            continue;
        };
        let dataset = if let Some(s) = obj.get("dataset").and_then(Value::as_str).filter(|s| !s.is_empty()) {
            s.to_string()
        } else {
            logger.log(&format!("missing or empty dataset for uid={} — skipping", uid));
            continue;
        };
        // parse prompt_count as either string or number
        let pc_str = if let Some(s) = obj.get("prompt_count").and_then(Value::as_str) {
            s.to_string()
        } else if let Some(n) = obj.get("prompt_count").and_then(Value::as_u64) {
            n.to_string()
        } else {
            logger.log(&format!("invalid prompt_count for uid={}", uid));
            continue;
        };
        let prompt_count: u32 = match pc_str.parse() {
            Ok(n) => n,
            Err(_) => {
                logger.log(&format!("invalid prompt_count '{}' for uid={} — skipping", pc_str, uid));
                continue;
            }
        };

        // collect all instruct_* and original keys into extra
        let mut extra = JsonMap::new();
        for (k, val) in obj.iter() {
            if k.starts_with("instruct_") || k == "instruction_original" {
                extra.insert(k.clone(), val.clone());
            }
        }

        map.insert(
            uid.clone(),
            AnsRecord {
                _uid:         uid.clone(),
                dataset,
                prompt_count,
                extra,
            },
        );
    }
    map
}

// New struct to hold an “answer” entry
#[derive(Debug, Clone)]
struct AnsRecord {
    _uid:         String,
    dataset:      String,
    prompt_count: u32,
    extra:        JsonMap<String, Value>,
}


// Main
#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.datasets.len() != cli.instructions.len() {
        return Err(anyhow!("--datasets and --instructions must have the same length"));
    }

    // logging
    fs::create_dir_all("logs")?;
    let ts = Local::now().format("%Y%m%d-%H%M%S");
    let log_path = Path::new("logs")
        .join(format!("{}_{}_{}.log", cli.log_name, cli.output.file_stem().unwrap().to_string_lossy(), ts));
    let mut logger = Logger::new(&log_path)?;
    logger.log(&format!(
        "run started – model={} margin={} api_cap={} ",
        cli.model, cli.margin, cli.api_call_max
    ));

    // I/O
    let mut instr_maps: HashMap<String, HashMap<String, Record>> = HashMap::new();
    for (path, ds) in cli.instructions.iter().zip(cli.datasets.iter()) {
        let map = read_records(path, &mut logger);
        instr_maps.insert(ds.clone(), map);
    }
    let ans_map = read_answers(&cli.answers, &mut logger);

    // ensure output directory exists BEFORE loading existing scores
    if let Some(parent) = cli.output.parent() {
        fs::create_dir_all(parent)
            .context("failed to create output directory")?;
    }
    // validate that all instruction maps loaded and answers exist
    if instr_maps.values().any(|m| m.is_empty()) || ans_map.is_empty() {
        return Err(anyhow!("instruction or answer JSON could not be read"));
    }

    // load / init existing scores (keyed by uid now)
    let mut scored: HashMap<String, JsonMap<String, Value>> = if cli.output.exists() {
        logger.log("loading existing output for resume mode");
        serde_json::from_str::<Vec<JsonMap<String, Value>>>(&fs::read_to_string(&cli.output)?)?
            .into_iter()
            .map(|obj| {
                let uid = obj
                    .get("uid")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("missing uid in existing output"))?
                    .to_string();
                Ok((uid, obj))
            })
            .collect::<Result<HashMap<String, JsonMap<String, Value>>, anyhow::Error>>()?
    } else {
        HashMap::new()
    };

    // HTTP client
    let api_key = cli
        .api_key
        .clone()
        .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .context("provide --api-key or set GOOGLE_API_KEY")?;
    let client = build_client()?;
    let mut api_calls_used = 0u32;

    // context limit for this model
    let ctx_limit = *MODEL_LIMITS
        .get(cli.model.as_str())
        .ok_or_else(|| anyhow!("unknown model {}", cli.model))?;

    // preparation: sort by uid
    let mut ans_sorted: Vec<(&String, &AnsRecord)> = ans_map.iter().collect();
    //ans_sorted.sort_by_key(|(uid, _)| uid.clone());
    ans_sorted.sort_by_key(|(uid, _)| *uid);

    // progress bar
    let bar = ProgressBar::new(ans_sorted.len() as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) {eta}"
        )
        .expect("invalid progress bar template"),
    );

    // main loop: iterate over each answer and look up its instruction
    for (uid, ans) in ans_sorted {
        // find the right instruction map for this dataset
        let instr_map = match instr_maps.get(&ans.dataset) {
            Some(m) => m,
            None => {
                logger.log(&format!("no dataset map for {} – skipping", ans.dataset));
                bar.inc(1);
                continue;
            }
        };
        // convert prompt_count back to string for lookup
        let pc_key = ans.prompt_count.to_string();
        let inst = match instr_map.get(&pc_key) {
            Some(r) => r,
            None => {
                logger.log(&format!("no instruction for {} → skip", uid));
                bar.inc(1);
                continue;
            }
        };

        // already_done logic, but now keyed by uid:
        let already_done: HashSet<String> = scored
            .get(uid)
            .map(|obj| {
                obj.keys() 
                    .filter(|k| k.starts_with("instruct_") || *k == "instruction_original")
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        // collect pending keys from ans.extra
        let pending: Vec<String> = {
            let mut keys = Vec::new();
            if ans.extra.contains_key("instruction_original") {
                keys.push("instruction_original".to_string());
            }
            keys.extend(
                ans.extra
                    .keys()
                    .filter(|k| k.starts_with("instruct_"))
                    .cloned(),
            );
            keys.into_iter()
                .filter(|k| !already_done.contains(k))
                .collect()
        };
        if pending.is_empty() {
            bar.inc(1);
            continue;
        }

        // chunking
        let mut cursor = 0usize;
        while cursor < pending.len() {
            if api_calls_used >= cli.api_call_max {
                logger.log("API cap reached → aborting early");
                break;
            }

            // greedy accumulate
            let mut chunk: Vec<String> = Vec::new();
            let mut section = String::new();
            let mut est_tokens = 0usize;
            while cursor < pending.len()
                && chunk.len() < cli.chunk_max
            {
                let key = &pending[cursor];
                
                // build paraphrase block
                let instr_text = if key == "instruction_original" {
                    &inst.instruction_original
                } else {
                    inst.extra.get(key).and_then(Value::as_str).unwrap_or("")
                };

                // pull answer text from ans.extra
                let ans_text_raw = ans.extra
                    .get(key)
                    .and_then(Value::as_str)
                    .unwrap_or("");

                // strip boiler-plate that doesn’t matter for quality but eats tokens
                let ans_text = ans_text_raw
                    .trim_start_matches("!?\n\n### Response:\n")
                    .trim_start_matches("?\n\n### Response:\n")
                    .trim_start_matches(".\n\n### Response:\n")
                    .trim_start_matches("### Response:\n")
                    .trim_start_matches("Response:\n")
                    .trim_start_matches("Response\n")
                    .trim();

                if ans_text.is_empty() {
                    logger.log(&format!("uid={} key={} has no answer – skipped", uid, key));
                    cursor += 1;
                    continue;
                }

                let block = format!(
                    "### {key}\n[Instruction]\n{instr_text}\n\n[Answer]\n{ans_text}\n\n"
                );
                let block_tokens = estimate_tokens(&block);

                if est_tokens + block_tokens + PROMPT_PREAMBLE_TOKENS >= ctx_limit - cli.margin {
                    break; // would overflow
                }
                if chunk.is_empty()
                    && block_tokens + PROMPT_PREAMBLE_TOKENS >= ctx_limit - cli.margin
                {
                    logger.log(&format!("key {key} is too large – skipped"));
                    cursor += 1;
                    continue;
                }
                section.push_str(&block);
                est_tokens += block_tokens;
                chunk.push(key.clone());
                cursor += 1;
            }

            if chunk.is_empty() {
                continue;
            }

            logger.log(&format!(
                "uid={} – chunk {} keys  est_tokens={}  ctx_limit={}  margin={}",
                uid,
                chunk.len(),
                est_tokens,
                ctx_limit,
                cli.margin
            ));

            let prompt = build_eval_prompt(&section);
            if DEBUG_IDS.contains(&inst.prompt_count) {
                let dump = format!("logs/debug_prompt_{}_chunk{}.txt", uid, chunk.len());
                fs::write(&dump, &prompt)?;
                logger.log(&format!("debug prompt written -> {dump}"));
            }

            let mut success = false;
            let mut attempt_used = 0;
            for attempt in 1..=cli.max_attempts {
                attempt_used = attempt;
                logger.log(&format!("API call (attempt {attempt}/{})", cli.max_attempts));
                match query_gemini(&client, &api_key, &cli.model, &prompt).await {
                    Ok(obj) => {
                        success = true;
                        // merge result under uid
                        let entry = scored.entry(uid.clone()).or_insert_with(|| {
                            let mut base = JsonMap::new();
                            base.insert("prompt_count".into(), json!(inst.prompt_count));
                            base
                        });
                        for key in &chunk {
                            if let Some(v) = obj.get(key) {
                                entry.insert(key.clone(), v.clone());
                            } else {
                                logger.log(&format!("missing key {key} in response"));
                            }
                        }
                        break;
                    }
                    Err(e) if attempt < cli.max_attempts => {
                        logger.log(&format!("attempt {attempt} failed: {e}"));
                        let backoff = 500u64 * 2u64.pow(attempt as u32);
                        sleep(Duration::from_millis(backoff)).await;
                    }
                    Err(e) => {
                        logger.log(&format!("all attempts failed for chunk: {e}"));
                    }
                }
            }
            api_calls_used += attempt_used as u32;
            if cli.delay_ms > 0 && success && attempt_used == 1 {
                sleep(Duration::from_millis(cli.delay_ms)).await;
            }
        }

        bar.inc(1);
    }
    bar.finish();

    // save
    let mut vec_out: Vec<JsonMap<String, Value>> = scored
        .into_iter()
        .map(|(_, v)| v)
        .collect();
    vec_out.sort_by_key(|m| m.get("prompt_count").and_then(Value::as_u64).unwrap_or(0));
    fs::write(&cli.output, serde_json::to_string_pretty(&vec_out)?)?;
    logger.log("results written");

    println!("finished – log at {}", log_path.display());
    Ok(())
}

// HTTP & prompt
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

Return *only* valid JSON (no markdown, no code fences) where each key maps to an array of ten integers 0-10.
The returned JSON must confirm to this Pydantic model:

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
    prompt: &str,
) -> Result<JsonMap<String, Value>> {
    let url = format!("{ENDPOINT}/models/{model}:generateContent?key={key}");
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
    let json_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| anyhow!("unexpected response structure"))?;
    Ok(serde_json::from_str(json_text.trim())?)
}
