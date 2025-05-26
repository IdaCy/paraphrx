/*
cargo run \
  --manifest-path c_assess_inf/Cargo.toml \
  --release -- \
  --version-set politeness \
  b_tests/data/alpaca_10_politeness.json \
  c_assess_inf/output/results_all.json \
  c_assess_inf/output/results_all_eval.json
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
    // STYLE / TONE
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
    output: String,                    // the canonical / gold answer
    #[serde(flatten)]
    extra: JsonMap<String, Value>,     // paraphrases + answers + evals
}

// command-line args-
#[derive(Parser, Debug)]
#[command(version, author, about = "Assess paraphrase answers with Gemini")]
struct Cli {
    /// Gold file that still contains the authoritative `output`
    gold: PathBuf,
    /// File that already contains the paraphrase answers
    answers: PathBuf,
    /// File to write evaluation to
    output: PathBuf,

    /// Which key-set to evaluate (style | length | obstruction | ...)
    #[arg(long, default_value = "style")]
    version_set: String,

    /// Gemini call retries
    #[arg(long, default_value_t = 3)]
    max_attempts: u8,
}

const MODEL: &str = "gemini-2.5-flash-preview-04-17";
const ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";

//-- main------
#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // gold --
    let gold_raw = fs::read_to_string(&cli.gold)
        .with_context(|| format!("failed to read {}", cli.gold.display()))?;
    let gold_map: HashMap<String, Record> = serde_json::from_str::<Vec<Record>>(&gold_raw)?
        .into_iter()
        .map(|r| (r.prompt_count.to_string(), r))
        .collect();

    //-------------------------- answers file --
    let ans_raw = fs::read_to_string(&cli.answers)
        .with_context(|| format!("failed to read {}", cli.answers.display()))?;
    let mut records: Vec<Record> = serde_json::from_str(&ans_raw)?;

    // misc --
    let keys = VERSION_SETS
        .get(cli.version_set.as_str())
        .ok_or_else(|| anyhow!("unknown version set {}", cli.version_set))?;
    let schema = schema_for();

    let api_key = env::var("GOOGLE_API_KEY").context("GOOGLE_API_KEY not set")?;
    let client = build_client()?;

    // prog bar --
    let bar = ProgressBar::new(records.len() as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
             {pos}/{len} ({eta})",
        )
        .unwrap(),
    );

    // loop --
    for rec in &mut records {
        let gold = gold_map
            .get(&rec.prompt_count.to_string())
            .ok_or_else(|| anyhow!("gold missing id {}", rec.prompt_count))?
            .clone();

        for &key in *keys {
            // The inference step should have produced e.g. `instruct_rude_answer`
            let answer_key = format!("{key}_answer");
            let Some(Value::String(answer)) = rec.extra.get(&answer_key) else {
                return Err(anyhow!(
                    "record {} missing candidate answer for {answer_key}",
                    rec.prompt_count
                ));
            };

            // Skip if we already evaluated
            let eval_key = format!("{key}_eval");
            if rec.extra.contains_key(&eval_key) {
                continue;
            }

            let prompt = build_eval_prompt(
                &rec.instruction_original,
                &gold.output,
                answer,
            );

            let mut success = false;
            for attempt in 1..=cli.max_attempts {
                match query_gemini(&client, &api_key, &schema, prompt.clone()).await {
                    Ok(eval_obj) => {
                        rec.extra
                            .insert(eval_key.clone(), Value::Object(eval_obj));
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

    / write out --
    fs::write(&cli.output, serde_json::to_string_pretty(&records)?)?;
    println!("evaluation written to {}", cli.output.display());
    Ok(())
}

/// ------------------------- helpers & utilities
fn build_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Ok(reqwest::Client::builder().default_headers(headers).build()?)
}

/// Prompt for grading a *single* candidate answer
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

/// Fixed JSON schema for the evaluation object
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

/// Send the grading request to Gemini
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
