/*
cargo run \
  --manifest-path c_assess_inf/Cargo.toml \
  --release -- \
  b_tests/data/alpaca_2_politeness.json \
  b_tests/assess_inf/results_2.json \
  b_tests/assess_inf/results_2_eval2.json
*/

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value};
use std::{
    collections::HashMap,
    env, fs,
    path::PathBuf,
};
use tokio::time::{sleep, Duration};

// data structs
#[derive(Debug, Deserialize, Serialize, Clone)]
struct Record {
    prompt_id: String,
    prompt_count: u32,
    #[serde(alias = "instruction", alias = "instruction_original")]
    instruction_original: String,

    #[serde(default)]
    output: Option<String>,

    #[serde(flatten)]
    extra: JsonMap<String, Value>,
}

// command-line args
#[derive(Parser, Debug)]
#[command(version, author, about = "Assess paraphrase answers with Gemini")]
struct Cli {
    instructions: PathBuf,
    answers: PathBuf,
    output: PathBuf,

    #[arg(long, default_value_t = 3)]
    max_attempts: u8,
}

const MODEL: &str = "gemini-2.5-flash-preview-04-17";
const ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";

// main
#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // instructions
    let instr_raw = fs::read_to_string(&cli.instructions)
        .with_context(|| format!("failed to read {}", cli.instructions.display()))?;
    let instr_map: HashMap<String, Record> = serde_json::from_str::<Vec<Record>>(&instr_raw)?
        .into_iter()
        .map(|r| (r.prompt_count.to_string(), r))
        .collect();

    // answers
    let ans_raw = fs::read_to_string(&cli.answers)
        .with_context(|| format!("failed to read {}", cli.answers.display()))?;
    let ans_map: HashMap<String, Record> = serde_json::from_str::<Vec<Record>>(&ans_raw)?
        .into_iter()
        .map(|r| (r.prompt_count.to_string(), r))
        .collect();

    // misc
    let api_key = env::var("GOOGLE_API_KEY").context("GOOGLE_API_KEY not set")?;
    let client = build_client()?;

    // progress bar
    let bar = ProgressBar::new(instr_map.len() as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
             {pos}/{len} ({eta})",
        )
        .unwrap(),
    );

    let mut results: Vec<Value> = Vec::new();

    // loop through prompts
    for (id, inst) in &instr_map {
        let ans = ans_map
            .get(id)
            .ok_or_else(|| anyhow!("answers missing id {}", id))?;

        // gather keys: instruction_original + any instruct_* in instruction record
        let mut keys: Vec<String> = vec!["instruction_original".to_string()];
        keys.extend(
            inst.extra
                .keys()
                .filter(|k| k.starts_with("instruct_"))
                .cloned(),
        );
        keys.sort();
        keys.dedup();

        // assemble variant section
        let mut section = String::new();
        for key in &keys {
            let instr_text = if key == "instruction_original" {
                inst.instruction_original.as_str()
            } else {
                inst.extra
                    .get(key)
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("instruction missing {key} for id {id}"))?
            };

            let ans_text = if key == "instruction_original" {
                ans.instruction_original.as_str()
            } else {
                ans.extra
                    .get(key)
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("answer missing {key} for id {id}"))?
            };

            section.push_str(&format!(
                "### {key}\n[Instruction]\n{instr_text}\n\n[Answer]\n{ans_text}\n\n"
            ));
        }

        let prompt = build_eval_prompt(&section);

        let mut eval_json: JsonMap<String, Value> = JsonMap::new();
        let mut success = false;

        for attempt in 1..=cli.max_attempts {
            match query_gemini(&client, &api_key, prompt.clone()).await {
                Ok(obj) => {
                    eval_json = obj;
                    success = true;
                    break;
                }
                Err(e) if attempt < cli.max_attempts => {
                    eprintln!(
                        "[warn] id {id} attempt {}/{} failed: {}",
                        attempt, cli.max_attempts, e
                    );
                    sleep(Duration::from_millis(300 * attempt as u64)).await;
                }
                Err(e) => return Err(anyhow!("id {id}: {e}")),
            }
        }

        if !success {
            return Err(anyhow!("evaluation failed for id {}", id));
        }

        // build result object
        let mut res_obj = JsonMap::new();
        res_obj.insert("prompt_id".to_string(), Value::String(inst.prompt_id.clone()));
        res_obj.insert(
            "prompt_count".to_string(),
            serde_json::to_value(inst.prompt_count)?,
        );

        for key in &keys {
            let arr_val = eval_json
                .get(key)
                .cloned()
                .ok_or_else(|| anyhow!("evaluation missing key {key} for id {id}"))?;
            res_obj.insert(key.clone(), arr_val);
        }

        results.push(Value::Object(res_obj));
        bar.inc(1);
    }

    bar.finish_with_message("done");

    // write out
    fs::write(&cli.output, serde_json::to_string_pretty(&results)?)?;
    println!("evaluation written to {}", cli.output.display());
    Ok(())
}

// helpers
fn build_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Ok(reqwest::Client::builder().default_headers(headers).build()?)
}

fn build_eval_prompt(variant_section: &str) -> String {
    format!(
r#"You are an expert evaluator.

For every answer below, assess it against **ten metrics**. Each metric must be scored on a 0–5 integer scale (higher is better).

Metrics (use **exact** order):
1. Task Fulfilment / Relevance
2. Usefulness & Actionability
3. Factual Accuracy & Verifiability
4. Efficiency / Depth & Completeness
5. Reasoning Quality / Transparency
6. Tone & Likeability
7. Adaptation to Context
8. Safety & Bias Avoidance
9. Structure & Formatting & UX Extras
10. Creativity

Return **only** JSON conforming to this Pydantic model (nothing else, no code fences):

```python
from typing import Dict, List
from pydantic import BaseModel, conlist

class EvalResult(BaseModel):
    __root__: Dict[str, conlist(int, min_items=10, max_items=10)]
```

Begin data to evaluate:

{variant_section}
"#)
}

async fn query_gemini(
    client: &reqwest::Client,
    key: &str,
    prompt: String,
) -> Result<JsonMap<String, Value>> {
    let url = format!(
        "{ENDPOINT}/models/{MODEL}:generateContent?key={key}",
        ENDPOINT = ENDPOINT,
        MODEL    = MODEL,
        key      = key
    );
    let body = json!({
        "contents":[{ "role":"user","parts":[{ "text": prompt }] }],
        "generationConfig":{
            "responseMimeType":"application/json"
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
