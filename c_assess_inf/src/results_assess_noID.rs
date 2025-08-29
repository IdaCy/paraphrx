/*
cargo results_assess \
  --model gemini-2.0-flash \
  --run-name my_alpaca_run \
  a_data/alpaca/merge_instructs/all.json \
  c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/instruct_merged/all.json \
  c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/instruct_merged/all_results.json
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

    // A name for the run, prepended to log and issues files
    #[arg(long)]
    run_name: Option<String>,

    // Gemini model name (e.g. gemini-2.5-flash-preview-05-20)
    #[arg(long, default_value = "gemini-2.0-flash")]
    model: String,

    #[arg(long, default_value_t = 5)]
    max_attempts: u8,

    /// Hard cap on number of API calls this run (default: 200)
    #[arg(long = "max-calls", default_value_t = 109)]
    max_calls: usize,

    // Milliseconds to wait after every successful request (avoid 429s)
    #[arg(long, default_value_t = 200)]
    delay_ms: u64,

    // Google API key (overrides $GOOGLE_API_KEY)
    #[arg(long = "api-key", value_name = "KEY")]
    api_key: Option<String>,

    /// Max paraphrases (keys) per single LLM request (always includes instruction_original)
    #[arg(long = "batch-size", default_value_t = 25)]
    batch_size: usize,
}

// Minimal schema with non-empty properties and required items for arrays
fn schema_for_keys(keys: &[String]) -> serde_json::Value {
    let mut props = serde_json::Map::new();
    for k in keys {
        props.insert(
            k.clone(),
            json!({
                "type": "array",
                "items": { "type": "integer" }
            })
        );
    }
    json!({
        "type": "object",
        "properties": props
    })
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
    
    // Prepend run_name to log and issue files if provided
    let run_name_prefix = cli.run_name.as_deref().map(|n| format!("{}_", n)).unwrap_or_default();

    // logs/<run_name>_<stem>_<timestamp>.logs
    let log_path = log_dir.join(format!("{}{}_{}.logs", run_name_prefix, stem, ts));

    let mut logger = Logger::new(&log_path)?;
    logger.log(&format!("run started -> model={} log={}", cli.model, log_path.display()));

    // I/O
    logger.log("reading json files");
    let instr_map = read_records(&cli.instructions, &mut logger);
    let ans_map   = read_records(&cli.answers,     &mut logger);

    // Load existing results to allow resuming
    let mut results: Vec<Value> = if cli.output.exists() {
        fs::read_to_string(&cli.output)
            .and_then(|s| serde_json::from_str(&s).map_err(Into::into))
            .unwrap_or_else(|e| {
                logger.log(&format!("[warn] could not parse existing results from {}: {}. Starting fresh.", cli.output.display(), e));
                Vec::new()
            })
    } else {
        Vec::new()
    };
    let processed_ids: HashSet<u32> = results.iter()
        .filter_map(|v| v.get("prompt_count").and_then(Value::as_u64).map(|pc| pc as u32))
        .collect();

    let api_key = cli
        .api_key
        .clone()
        .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .context("provide --api-key or set GOOGLE_API_KEY")?;
    let client  = build_client()?;

    // sort so we run strictly in prompt_count order
    let mut instr_sorted: Vec<(&String, &Record)> = instr_map.iter().collect();
    instr_sorted.sort_by_key(|(_, r)| r.prompt_count);

    // Skip items already processed
    let tasks_remaining: Vec<_> = instr_sorted
        .into_iter()
        .filter(|(_, r)| !processed_ids.contains(&r.prompt_count))
        .collect();

    let remaining_unprocessed_len = tasks_remaining.len();

    let mut skipped_missing_answers: usize = 0;
    let mut missing_ids: Vec<String> = Vec::new();

    let tasks_with_answers: Vec<_> = tasks_remaining
        .into_iter()
        .filter(|(id, _)| {
            if ans_map.contains_key(*id) { true } else {
                skipped_missing_answers += 1;
                missing_ids.push((*id).clone());
                false
            }
        })
        .collect();

    let mut issues = Vec::new();
    issues.extend(missing_ids.into_iter().map(|id| format!("answers missing id {id}")));

    if skipped_missing_answers > 0 {
        logger.log(&format!(
            "Skipping {} items with no matching answers.", skipped_missing_answers
        ));
    }

    // Batch-aware accounting: cap counts *API calls* (batches), not items
    logger.log(&format!(
        "Total instructions: {}, Already processed: {}, Remaining (unprocessed): {}, With answers: {}. max_calls={} (counts batches).",
        instr_map.len(),
        processed_ids.len(),
        remaining_unprocessed_len,
        tasks_with_answers.len(),
        cli.max_calls
    ));

    let mut calls_made: usize = 0;
    if tasks_with_answers.is_empty() {
        println!("Nothing to do (either all processed or missing answers). Log: {}", log_path.display());
        return Ok(());
    }

    let bar = ProgressBar::new(tasks_with_answers.len() as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )?,
    );

    for (idx, (id, inst)) in tasks_with_answers.into_iter().enumerate() {
        if calls_made >= cli.max_calls {
            logger.log(&format!("Reached max-calls cap ({}) — stopping cleanly.", cli.max_calls));
            break;
        }
        let progress = format!("({}/{})", idx + 1, bar.length().unwrap_or(0));
        logger.log(&format!("▶ start id {id} {progress}"));

        let (attempts_used_overall, processed_this_run, calls_used_for_id) = match process_single(
            id.as_str(), inst, &ans_map, &client, &api_key, &cli.model,
            cli.max_attempts, cli.batch_size, cli.delay_ms, &mut logger, &mut results, &mut issues,
        ).await {
            Ok(t) => t,
            Err(e) => {
                let msg = e.to_string();
                logger.log(&format!("[error] id {id}: {msg}"));
                // If this was our sentinel "rate limit exhausted", save and quit
                if msg.contains("QUIT_RATE_LIMIT_AFTER_429x3") {
                    // Always save best-effort state before quitting
                    let _ = fs::write(&cli.output, serde_json::to_string_pretty(&results)?);
                    // Write issues gathered so far
                    if !issues.is_empty() {
                        let stem = cli
                            .output
                            .file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned();
                        let run_name_prefix = cli.run_name.as_deref().map(|n| format!("{}_", n)).unwrap_or_default();
                        let issues_path = cli.output.with_file_name(format!("{}{}.issues.json", run_name_prefix, stem));
                        let _ = fs::write(&issues_path, serde_json::to_string_pretty(&issues)?);
                    }
                    logger.log("[fatal] 3×429 encountered; progress saved. Quitting.");
                    println!("hit API rate limit repeatedly (3×429). Saved progress to {}. Log: {}", cli.output.display(), log_path.display());
                    return Ok(());
                }
                // If this was our sentinel for INTERNAL 500s, save and quit
                if msg.contains("QUIT_INTERNAL_ERROR_AFTER_500x3") {
                    // Always save best-effort state before quitting
                    let _ = fs::write(&cli.output, serde_json::to_string_pretty(&results)?);
                    // Write issues gathered so far
                    if !issues.is_empty() {
                        let stem = cli
                            .output
                            .file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned();
                        let run_name_prefix = cli.run_name.as_deref().map(|n| format!("{}_", n)).unwrap_or_default();
                        let issues_path = cli.output.with_file_name(format!("{}{}.issues.json", run_name_prefix, stem));
                        let _ = fs::write(&issues_path, serde_json::to_string_pretty(&issues)?);
                    }
                    logger.log("[fatal] 3×500 INTERNAL encountered; progress saved. Quitting.");
                    println!("hit 3×500 INTERNAL errors. Saved progress to {}. Log: {}", cli.output.display(), log_path.display());
                    return Ok(());
                }
                issues.push(format!("id {id}: {msg}"));
                // treat as 'slow' so we skip sleep on this item
                (cli.max_attempts, false, 0)
            }
        };
        calls_made += calls_used_for_id;
        bar.inc(1);
        // Global rate-limit pause (configurable via --delay-ms). Pause only if flew through on first go.
        if cli.delay_ms > 0 && attempts_used_overall == 1 {
            sleep(Duration::from_millis(cli.delay_ms)).await;
        }
        if processed_this_run {
            if let Err(e) = fs::write(&cli.output, serde_json::to_string_pretty(&results)?) {
                logger.log(&format!("[error] id {id}: Failed to save intermediate results: {e}"));
            }
        }
        if calls_made >= cli.max_calls {
            logger.log(&format!("Reached max-calls cap ({}) — stopping cleanly.", cli.max_calls));
            break;
        }
    }
    bar.finish_with_message("done");

    logger.log("run finished, results are up-to-date");

    if !issues.is_empty() {
        // Use run_name_prefix for issues file as well
        let issues_path = cli.output.with_file_name(format!("{}{}.issues.json", run_name_prefix, stem));
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

// Batching helpers – place these *above* process_single
fn chunk_keys(keys: &[String], batch_size: usize) -> Vec<Vec<String>> {
    // Ensure stable, deterministic order
    let mut ks = keys.to_owned();
    ks.sort();

    // Always include instruction_original in every batch as an anchor/baseline
    let anchor = "instruction_original".to_string();
    let mut rest: Vec<String> = ks.into_iter().filter(|k| *k != anchor).collect();

    // If batch_size is tiny, still force room for the anchor
    let per_batch_rest = batch_size.saturating_sub(1).max(1);

    let mut out = Vec::new();
    for chunk in rest.chunks(per_batch_rest) {
        let mut v = Vec::with_capacity(chunk.len() + 1);
        v.push(anchor.clone());
        v.extend(chunk.iter().cloned());
        out.push(v);
    }
    if out.is_empty() {
        out.push(vec![anchor]);
    }
    out
}

fn build_section_for_keys(
    inst: &Record,
    ans: &Record,
    keys: &[String],
)->String{
    let input_opt = inst.extra.get("input").and_then(Value::as_str).map(str::trim);
    let mut section = String::new();
    for key in keys {
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

        section.push_str(&format!("### {key}\n[Instruction]\n{instr}\n\n"));
        if let Some(inp) = input_opt {
            if !inp.is_empty() {
                section.push_str(&format!("[Input]\n{}\n\n", inp));
            }
        }
        section.push_str(&format!("[Answer]\n{}\n\n", ans_txt));
    }
    section
}

// Validate we got exactly 10 integers
fn is_ten_ints(v: &Value) -> bool {
    match v.as_array() {
        Some(a) if a.len() == 10 => a.iter().all(|x| x.as_i64().is_some()),
        _ => false,
    }
}

async fn process_single(
    id: &str,
    inst: &Record,
    ans_map: &HashMap<String, Record>,
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    max_attempts: u8,
    batch_size: usize,
    delay_between_batches_ms: u64,
    logger: &mut Logger,
    results: &mut Vec<Value>,
    issues: &mut Vec<String>,
) -> Result<(u8, bool, usize)> { // (attempts_used_overall, processed, api_calls_used)
    let ans = match ans_map.get(id) {
        Some(a) => a,
        None => {
            issues.push(format!("answers missing id {id}"));
            return Ok((max_attempts, false, 0));
        }
    };
    // Collect keys: original + all instruct_* present in either file
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

    // Make fixed-size batches that always include instruction_original
    let batches = chunk_keys(&keys, batch_size.max(1));

    let mut eval_json_all = JsonMap::new();
    let mut attempts_used_overall: u8 = 1;
    let mut api_calls_used: usize = 0;

    for (bix, batch_keys) in batches.iter().enumerate() {
        let section = build_section_for_keys(inst, ans, batch_keys);
        if section.len() > 95_000 {
            issues.push(format!("id {id}: prompt too large for batch {} ({} bytes)", bix + 1, section.len()));
            return Ok((max_attempts, false, api_calls_used));
        }

        let schema = schema_for_keys(batch_keys);
        let prompt = build_eval_prompt(&section, batch_keys);

        let mut success = false;
        let mut eval_json_one = JsonMap::new();
        let mut rate_limit_hits: u8 = 0;
        let mut internal_hits: u8 = 0;

        for attempt in 1..=max_attempts {
            logger.log(&format!("[call] id {id} batch {}/{} attempt {}/{}", bix + 1, batches.len(), attempt, max_attempts));
            match query_gemini(client, api_key, model, schema.clone(), prompt.clone()).await {
                Ok(obj) => {
                    logger.log(&format!("[ok]   id {id} batch {}/{} attempt {}/{}", bix + 1, batches.len(), attempt, max_attempts));
                    // Start with what we got
                    let mut got = obj;

                    // Identify missing or malformed keys
                    let mut todo: Vec<String> = batch_keys
                        .iter()
                        .filter(|k| match got.get(*k) {
                            Some(v) => !is_ten_ints(v),
                            None => true,
                        })
                        .cloned()
                        .collect();

                    // Try up to 2 recovery rounds querying only the missing/bad keys
                    let mut recovery_round = 0;
                    while !todo.is_empty() && recovery_round < 2 {
                        recovery_round += 1;
                        logger.log(&format!(
                            "[info] id {id} batch {}/{}: recovering {} key(s) (round {})",
                            bix + 1, batches.len(), todo.len(), recovery_round
                        ));
                        let section_retry = build_section_for_keys(inst, ans, &todo);
                        // guard prompt size in recovery too
                        if section_retry.len() > 95_000 {
                            issues.push(format!(
                                "id {id}: recovery prompt too large for batch {}/{} ({} bytes)",
                                bix + 1, batches.len(), section_retry.len()
                            ));
                            break;
                        }
                        let schema_retry = schema_for_keys(&todo);
                        let prompt_retry = build_eval_prompt(&section_retry, &todo);
                        match query_gemini(client, api_key, model, schema_retry, prompt_retry).await {
                            Ok(obj2) => {
                                api_calls_used += 1;
                                // merge new keys
                                for (k, v) in obj2.iter() {
                                    got.insert(k.clone(), v.clone());
                                }
                                // recompute todo
                                todo = todo
                                    .into_iter()
                                    .filter(|k| got.get(k).map(is_ten_ints).unwrap_or(false) == false)
                                    .collect();
                            }
                            Err(e2) => {
                                issues.push(format!(
                                    "id {id} batch {}/{}: recovery round {} failed: {}",
                                    bix + 1, batches.len(), recovery_round, e2
                                ));
                                break;
                            }
                        }
                    }

                    eval_json_one = got;
                    success = true;
                    attempts_used_overall = attempts_used_overall.max(attempt);
                    api_calls_used += 1; // initial successful call
                    // throttle between *successful* batches to avoid 429 QPS
                    if delay_between_batches_ms > 0 && (bix + 1) < batches.len() {
                        sleep(Duration::from_millis(delay_between_batches_ms)).await;
                    }
                    break;
                }
                Err(e) if e.to_string().contains("RATE_LIMIT_429") => {
                    rate_limit_hits += 1;
                    if rate_limit_hits >= 3 {
                        return Err(anyhow!("QUIT_RATE_LIMIT_AFTER_429x3"));
                    }
                    logger.log(&format!(
                        "[warn] id {id} batch {}/{} attempt {}/{}: 429 (hit {}/3) – pausing 60s",
                        bix + 1, batches.len(), attempt, max_attempts, rate_limit_hits
                    ));
                    sleep(Duration::from_secs(60)).await;
                }
                Err(e) if e.to_string().contains("500 Internal Server Error")
                    || e.to_string().contains("\"status\":\"INTERNAL\"") => {
                    internal_hits += 1;
                    if internal_hits >= 3 {
                        logger.log(&format!(
                            "[fatal] id {id} batch {}/{}: 3×500 INTERNAL — quitting run after saving progress.",
                            bix + 1, batches.len()
                        ));
                        // Signal to main() to save progress and exit the whole run
                        return Err(anyhow!("QUIT_INTERNAL_ERROR_AFTER_500x3"));
                    }
                    // Slightly stronger backoff for 500s than generic errors
                    let wait = 1_000u64 * 2u64.pow(attempt as u32)
                        + (SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .subsec_millis() as u64) % 500;
                    logger.log(&format!(
                        "[warn] id {id} batch {}/{} attempt {}/{}: 500 INTERNAL (hit {}/3) – backing off {}ms",
                        bix + 1, batches.len(), attempt, max_attempts, internal_hits, wait
                    ));
                    sleep(Duration::from_millis(wait)).await;
                }
                Err(e) if attempt < max_attempts => {
                    // backoff for non-429 errors
                    let wait = 500u64 * 2u64.pow(attempt as u32)
                        + (SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .subsec_millis() as u64) % 300;
                    logger.log(&format!(
                        "[warn] id {id} batch {}/{} attempt {}/{}: {}",
                        bix + 1, batches.len(), attempt, max_attempts, e
                    ));
                    sleep(Duration::from_millis(wait)).await;
                }
                Err(e) => {
                    issues.push(format!("id {id} batch {}/{}: {}", bix + 1, batches.len(), e));
                    return Ok((max_attempts, false, api_calls_used));
                }
            }
        }
        if !success {
            issues.push(format!("id {id}: all attempts failed for batch {}/{}", bix + 1, batches.len()));
            return Ok((max_attempts, false, api_calls_used));
        }

        // Merge this batch's keys
        for k in batch_keys {
            if let Some(v) = eval_json_one.get(k) {
                if is_ten_ints(v) {
                    eval_json_all.insert(k.clone(), v.clone());
                } else {
                    issues.push(format!(
                        "id {id}: bad shape for key {k} in batch {}/{}",
                        bix + 1, batches.len()
                    ));
                }
            } else {
                issues.push(format!("id {id}: missing eval key {k} in batch {}/{}", bix + 1, batches.len()));
            }
        }
    }

    // Write one row for the whole id once *all* batches succeeded
    let mut res_obj = JsonMap::new();
    res_obj.insert(
        "prompt_count".to_string(),
        serde_json::to_value(inst.prompt_count)?,
    );
    for k in &keys {
        if let Some(v) = eval_json_all.get(k) {
            res_obj.insert(k.clone(), v.clone());
        } else {
            issues.push(format!("id {id}: missing eval key {k} after merge"));
        }
    }
    results.push(Value::Object(res_obj));
    logger.log(&format!("[done] id {id} fully processed in {} batch call(s)", api_calls_used));
    Ok((attempts_used_overall, true, api_calls_used))
}

fn build_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Ok(reqwest::Client::builder().default_headers(headers).build()?)
}

fn build_eval_prompt(section: &str, keys: &[String]) -> String {
    let key_list = keys.join("\", \"");
    format!(r#"You are an expert evaluator.

For every answer below, assess it against **ten metrics**. Each metric must be scored on a 0-10 integer scale (higher is better).

Metrics (use **exact** order):
1. Task Fulfilment / Relevance
2. Usefulness & Actionability
3. Factual Accuracy & Verifiability
4. Efficiency / Depth & Completeness
5. Reasoning Quality / Transparency
6. Tone & Likeability
7. Adaptation to Context
8. Safety & Bias Avoidance
9. Structure & Formatting & UX Extras
10. Creativity

Return **only** JSON whose **top-level object has exactly these keys**:
["{key_list}"]
Each key maps to a list of **10 integers** (0–10) in the metric order above. No explanations, no extra keys.

Begin data to evaluate:

{section}
"#)
}

async fn query_gemini(
    client:&reqwest::Client,
    key:&str,
    model:&str,
    schema:Value,
    prompt:String,
)->Result<JsonMap<String,Value>>{
    // v1beta endpoint uses :generateContent
    let url = format!("{ENDPOINT}/models/{}:generateContent?key={}", model, key);
    let body = json!({
        "contents":[{"role":"user","parts":[{"text":prompt}]}],
        "generationConfig":{
            "temperature": 0.0,
            "topK": 1,
            "topP": 1.0,
            "responseMimeType":"application/json",
            "responseSchema": schema
        }
    });
    let resp=client.post(&url).json(&body).send().await?;
    if !resp.status().is_success(){
        if resp.status().as_u16() == 429 {
            return Err(anyhow!("RATE_LIMIT_429: {}", resp.text().await.unwrap_or_default()));
        } else {
            return Err(anyhow!("{} — {}",resp.status(),resp.text().await.unwrap_or_default()));
        }
    }
    let resp_json:Value=resp.json().await?;
    let json_text=resp_json["candidates"][0]["content"]["parts"][0]["text"].as_str().ok_or_else(||anyhow!("unexpected response structure"))?;
    Ok(serde_json::from_str(json_text.trim())?)
}
