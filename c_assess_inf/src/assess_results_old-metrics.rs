/*
cargo run \
  --manifest-path c_assess_inf/Cargo.toml \
  --release -- \
  --version-set politeness \
  b_tests/data/alpaca_2_politeness.json \
  b_tests/data/alpaca_2.json \
  b_tests/data/alpaca_2_eval2.json

*/

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use phf::phf_map;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value};
use std::{
    collections::HashMap,
    env, fs,
    path::PathBuf,
};
use tokio::time::{sleep, Duration};

// paraphrase key-sets
static VERSION_SETS: phf::Map<&'static str, &'static [&'static str]> = phf_map! {
    "politeness" => &[
        "instruction_original",
        "instruct_1_samelength",
        "instruct_2_polite",
        "instruct_3_properpolite",
        "instruct_4_superpolite",
        "instruct_5_longpolite"
    ]
};

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

// command-line args
#[derive(Parser, Debug)]
#[command(version, author, about = "Assess paraphrase answers with Gemini")]
struct Cli {
    gold:    PathBuf,
    answers: PathBuf,
    output:  PathBuf,

    #[arg(long, default_value = "style")]
    version_set: String,

    #[arg(long, default_value_t = 3)]
    max_attempts: u8,
}

const MODEL: &str = "gemini-2.5-flash-preview-04-17";
const ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";

// main
#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // gold
    let gold_raw = fs::read_to_string(&cli.gold)
        .with_context(|| format!("failed to read {}", cli.gold.display()))?;
    let gold_map: HashMap<String, Record> = serde_json::from_str::<Vec<Record>>(&gold_raw)?
        .into_iter()
        .map(|r| (r.prompt_count.to_string(), r))
        .collect();

    // answers
    let ans_raw = fs::read_to_string(&cli.answers)
        .with_context(|| format!("failed to read {}", cli.answers.display()))?;
    let mut records: Vec<Record> = serde_json::from_str(&ans_raw)?;

    // misc
    //let keys = VERSION_SETS
    //    .get(cli.version_set.as_str())
    //    .ok_or_else(|| anyhow!("unknown version set {}", cli.version_set))?;
    //let schema = schema_for();

    let predefined_keys = VERSION_SETS.get(cli.version_set.as_str()).copied();
    let schema = schema_for();

    let api_key = env::var("GOOGLE_API_KEY").context("GOOGLE_API_KEY not set")?;
    let client = build_client()?;

    // progress bar
    let bar = ProgressBar::new(records.len() as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
             {pos}/{len} ({eta})",
        )
        .unwrap(),
    );

    // loop
    for rec in &mut records {
        let gold = gold_map
            .get(&rec.prompt_count.to_string())
            .ok_or_else(|| anyhow!("gold missing id {}", rec.prompt_count))?
            .clone();

        let gold_output = gold
            .output
            .as_deref()
            .ok_or_else(|| anyhow!("gold missing output for id {}", rec.prompt_count))?;

        // gather *all* candidate keys:
        let mut keys: Vec<String> = Vec::new();

        // 1. any predefined set (still supported for backward compatibility)
        if let Some(set) = predefined_keys {
            keys.extend(set.iter().map(|s| s.to_string()));
        }

        // 2. everything that starts with “instruct_” in the record itself
        keys.extend(
            rec.extra
                .keys()
                .filter(|k| k.starts_with("instruct_") && !k.ends_with("_eval"))
                .cloned(),
        );

        // 3. always include the original
        keys.push("instruction_original".to_string());

        // deduplicate
        keys.sort();
        keys.dedup();

        for key in &keys {
            let key = key.as_str();          // &str for the rest of the code

            let answer: &str = if key == "instruction_original" {
                &rec.instruction_original
            } else {
                rec.extra
                    .get(key)
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!(
                        "record {} missing candidate answer for {}",
                        rec.prompt_count, key
                    ))?
            };

            let eval_key = format!("{key}_eval");
            if rec.extra.contains_key(&eval_key) {
                continue;
            }

            let prompt = build_eval_prompt(
                &gold.instruction_original,
                gold_output,
                answer,
            );

            let mut success = false;
            for attempt in 1..=cli.max_attempts {
                match query_gemini(&client, &api_key, &schema, prompt.clone()).await {
                    Ok(eval_obj) => {
                        rec.extra.insert(eval_key.clone(), Value::Object(eval_obj));
                        success = true;
                        break;
                    }
                    Err(e) if attempt < cli.max_attempts => {
                        eprintln!(
                            "[warn] id {} {key} attempt {}/{} failed: {}",
                            rec.prompt_count, attempt, cli.max_attempts, e
                        );
                        sleep(Duration::from_millis(300 * attempt as u64)).await;
                    }
                    Err(e) => return Err(anyhow!("id {}, {key}: {e}", rec.prompt_count)),
                }
            }

            if !success {
                return Err(anyhow!(
                    "evaluation failed for id {} key {key}",
                    rec.prompt_count
                ));
            }
        }
        bar.inc(1);
    }
    bar.finish_with_message("done");

    // write out
    fs::write(&cli.output, serde_json::to_string_pretty(&records)?)?;
    println!("evaluation written to {}", cli.output.display());
    Ok(())
}

// helpers
fn build_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Ok(reqwest::Client::builder().default_headers(headers).build()?)
}

fn build_eval_prompt(
    instruction: &str,
    gold_answer: &str,
    candidate: &str,
) -> String {
    format!(
r#"You are an expert evaluator.
We executed the *same* instruction twice:

[Instruction]
{instruction}

[Gold Answer] – authoritatively correct:
{gold_answer}

[Candidate Answer] – produced after paraphrasing the instruction:
{candidate}

Assess the candidate ONLY for faithfulness to the instruction, completeness, and correctness.
Return **one** JSON with exactly these keys:
* "is_correct" – boolean
* "score_0_to_5" – integer
* "explanation" – short (≤30 words) critique
No other keys, no prose."#
    )
}

fn schema_for() -> Value {
    json!({
        "type": "object",
        "properties": {
            "is_correct":  { "type": "boolean" },
            "score_0_to_5":{ "type": "integer", "minimum":0, "maximum":5 },
            "explanation": { "type": "string" }
        },
        "required": ["is_correct","score_0_to_5","explanation"]
    })
}

async fn query_gemini(
    client: &reqwest::Client,
    key: &str,
    schema: &Value,
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
            "responseMimeType":"application/json",
            "responseSchema": schema
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

    Ok(serde_json::from_str(json_text)?)
}
