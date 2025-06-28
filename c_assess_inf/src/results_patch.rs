/*
cargo results_patch \
  --instructions-dir a_data/alpaca/slice_500 \
  --answers-dir      c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/answers_slice_500 \
  --scores-dir       c_assess_inf/output/alpaca_answer_scores/gemma-2-2b-it \
  --issues-dir       c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/answers_slice_500_score_issues \
  --type style --type tone \
  --model "gemini-2.5-flash-preview-05-20" \
  --api-key "$GOOGLE_API_KEY"
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
use regex::Regex;
use std::ffi::OsStr;
//use std::borrow::Cow;

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

/// Re-score only the prompt-IDs that appear in <issues-dir>/<TYPE>_issues.json
async fn process_set(
    typ: &str,
    cli: &Cli,
    client: &reqwest::Client,
    api_key: &str,
    root_log: &mut Logger,
) -> Result<()> {
    // locate the four files that belong to this TYPE
    let instr_path  = cli.instructions_dir.join(format!("{typ}.json"));
    let ans_path    = cli.answers_dir.join(format!("{typ}.json"));
    let score_path = cli.scores_dir.join(format!("{typ}.json"));
    let issues_path = cli.issues_dir.join(format!("{typ}_issues.json"));

    // load JSON files we already have
    let repair_log_path = Path::new("logs").join(format!("{typ}_repair.log"));
    let mut logger = Logger::new(&repair_log_path)?;
    logger.log("loading files");

    let instr_map = read_records(&instr_path, &mut logger);
    let ans_map   = read_records(&ans_path,   &mut logger);

    // scores file may not exist on first run
    let mut scores_vec: Vec<Value> = if score_path.exists() {
        serde_json::from_str(&fs::read_to_string(&score_path)?)?
    } else { Vec::new() };

    // which prompt_count IDs still need a score?
    let issue_lines: Vec<String> = if issues_path.exists() {
        serde_json::from_str(&fs::read_to_string(&issues_path)?)?
    } else {
        logger.log("no issues file – nothing to repair");
        return Ok(());
    };

    let id_re = Regex::new(r"id (\d+):")?;
    let todo_ids: Vec<String> = issue_lines
        .iter()
        .filter_map(|l| id_re.captures(l).and_then(|c| c.get(1)).map(|m| m.as_str().to_string()))
        .collect();

    if todo_ids.is_empty() {
        logger.log("issues file contained no valid IDs");
        return Ok(());
    }
    logger.log(&format!("{} missing IDs", todo_ids.len()));

    // progress bar just for this set
    let bar = ProgressBar::new(todo_ids.len() as u64);
    bar.set_style(ProgressStyle::with_template(
        "{spinner:.green} {pos}/{len} {wide_bar:.cyan/blue} {elapsed_precise}",
    )?);

    // collect IDs that still fail
    let mut new_issues: Vec<String> = Vec::new();

    // evaluate each missing ID
    for id in &todo_ids {
        if let Some(instr_rec) = instr_map.get(id) {
            process_single(
                id,
                instr_rec,
                &ans_map,
                client,
                api_key,
                &cli.model,
                cli.max_attempts,
                &mut logger,
                &mut scores_vec,
                &mut new_issues,
            )
            .await?;
        } else {
            logger.log(&format!("id {id}: instructions missing – skipped"));
            new_issues.push(format!("id {id}: instructions missing"));
        }

        bar.inc(1);
        if cli.delay_ms > 0 {
            sleep(Duration::from_millis(cli.delay_ms)).await;
        }
    }
    bar.finish();

    // dump unresolved issues without touching the original file
    let issues_patched_path =
        issues_path.with_file_name(format!("{typ}_issues_patched.json"));
    fs::write(
        &issues_patched_path,
        serde_json::to_string_pretty(&new_issues)?,
    )?;

    // rename the original issues file so it will not be picked up next time
    let processed_path = issues_path.with_file_name(format!(
        "patch-processed_{}",
        issues_path
            .file_name()
            .and_then(OsStr::to_str)
            .unwrap_or_default()
    ));
    fs::rename(&issues_path, &processed_path)?;
    logger.log(&format!(
        "patched issues written → {} ({} remaining); original renamed → {}",
        issues_patched_path.display(),
        new_issues.len(),
        processed_path.display()
    ));

    // keep scores sorted by prompt_count and overwrite the file
    scores_vec.sort_by_key(|v| v["prompt_count"].as_u64().unwrap_or(0));

    // build “…_patched.json” next to the old file
    let patched_path = score_path.with_file_name(format!(
        "{}_patched.json",
        score_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
    ));

    fs::create_dir_all(&cli.scores_dir)?;
    fs::write(&patched_path, serde_json::to_string_pretty(&scores_vec)?)?;
    logger.log(&format!("patched scores written → {}", patched_path.display()));

    root_log.log(&format!("set '{typ}' patched; {} unresolved issues", new_issues.len()));
    Ok(())
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

#[derive(Parser, Debug)]
#[command(
    version,
    author,
    about = "Re-score failed items for one or more prompt sets"
)]
struct Cli {
    /// Folder that holds <TYPE>.json with the paraphrased instructions
    #[arg(long = "instructions-dir")]
    instructions_dir: PathBuf,

    /// Folder that holds <TYPE>.json with the model answers
    #[arg(long = "answers-dir")]
    answers_dir: PathBuf,

    /// Folder that holds <TYPE>.json with the numeric scores
    #[arg(long = "scores-dir")]
    scores_dir: PathBuf,

    /// Folder that holds <TYPE>_issues.json produced by the previous run
    #[arg(long = "issues-dir")]
    issues_dir: PathBuf,

    /// Which prompt sets to repair.  May be given several times:
    ///     --type style --type tone
    #[arg(long = "type", required = true, num_args = 1..)]
    types: Vec<String>,

    #[arg(long, default_value = "gemini-2.0-flash")]
    model: String,

    #[arg(long, default_value_t = 5)]
    max_attempts: u8,

    #[arg(long = "delay-ms", default_value_t = 200)]
    delay_ms: u64,

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

    // common setup (client + root log)
    let api_key = cli
        .api_key
        .clone()
        .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .context("provide --api-key or set GOOGLE_API_KEY")?;
    let client  = build_client()?;

    let log_dir = Path::new("logs");
    fs::create_dir_all(log_dir)?;
    let ts = Local::now().format("%Y%m%d-%H%M%S");
    let mut root_logger = Logger::new(log_dir.join(format!("rerun_{ts}.logs")))?;
    root_logger.log(&format!(
        "repair run – model={} – sets={:?}",
        cli.model, cli.types
    ));

    // process every <TYPE> that the user passed in
    for t in &cli.types {
        root_logger.log(&format!("── set '{t}' ──"));
        if let Err(e) = process_set(
            t,
            &cli,
            &client,
            &api_key,
            &mut root_logger,
        )
        .await
        {
            root_logger.log(&format!("[fatal] set '{t}': {e}"));
        }
    }

    println!("all done – see log files in {}", log_dir.display());
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

For every answer below, assess it against **ten metrics**. Each metric must be scored on a 0–10 integer scale (higher is better).

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
