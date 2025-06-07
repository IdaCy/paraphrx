/*
cargo run \
    --manifest-path a_data/preproc/rephras/Cargo.toml \
    --release -- \
    a_data/alpaca/alpaca_10_test.json \
    a_data/alpaca/alpaca_10_test_phrxed.json
*/

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{env, fs, path::PathBuf};
use tokio::time::{sleep, Duration};

// Pydantic‑style verification struct (serde handles validation)
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct Verification {
    instruct_1_samelength: String,
    instruct_2_polite: String,
    instruct_3_properpolite: String,
    instruct_4_superpolite: String,
    instruct_5_longpolite: String,
}

// Alpaca‑style record (only care about a few keys)
#[derive(Debug, Serialize, Deserialize)]
struct Record {
    prompt_count: u32,
    #[serde(alias = "instruction_original")] 
    instruction_original: String,
    #[serde(default)]
    instruct_1_samelength: Option<String>,
    #[serde(default)]
    instruct_2_polite: Option<String>,
    #[serde(default)]
    instruct_3_properpolite: Option<String>,
    #[serde(default)]
    instruct_4_superpolite: Option<String>,
    #[serde(default)]
    instruct_5_longpolite: Option<String>,
    // keep any extra fields intact
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Parser, Debug)]
#[command(version, author, about = "Generate graded‑polite paraphrases using Gemini & Rust")]
struct Cli {
    // Input Alpaca JSON file
    input: PathBuf,
    // Output JSON file
    output: PathBuf,
    // Maximum attempts per prompt if validation fails
    #[arg(long, default_value_t = 3)]
    max_attempts: u8,
}

const MODEL: &str = "gemini-2.5-flash-preview-04-17";
const ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Read dataset
    let data = fs::read_to_string(&cli.input)
        .with_context(|| format!("failed to read {}", cli.input.display()))?;
    let mut records: Vec<Record> = serde_json::from_str(&data)?;

    let key = env::var("GOOGLE_API_KEY").context("GOOGLE_API_KEY not set")?;
    let client = build_client()?;

    let bar = ProgressBar::new(records.len() as u64);
    bar.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap());

    for rec in &mut records {
        let prompt = build_prompt(&rec.instruction_original);
        let mut success = false;

        for attempt in 1..=cli.max_attempts {
            match query_gemini(&client, &key, prompt.clone()).await {
                Ok(ver) => {
                    rec.instruct_1_samelength = Some(ver.instruct_1_samelength);
                    rec.instruct_2_polite = Some(ver.instruct_2_polite);
                    rec.instruct_3_properpolite = Some(ver.instruct_3_properpolite);
                    rec.instruct_4_superpolite = Some(ver.instruct_4_superpolite);
                    rec.instruct_5_longpolite = Some(ver.instruct_5_longpolite);
                    success = true;
                    break;
                }
                Err(err) if attempt < cli.max_attempts => {
                    eprintln!(
                        "[warn] prompt_count {} attempt {}/{} failed: {}",
                        rec.prompt_count, attempt, cli.max_attempts, err
                    );
                    sleep(Duration::from_millis(500 * u64::from(attempt))).await;
                }
                Err(err) => return Err(anyhow!("id {}: {}", rec.prompt_count, err)),
            }
        }

        if !success {
            return Err(anyhow!("validation failed for prompt_count {}", rec.prompt_count));
        }
        bar.inc(1);
    }
    bar.finish_with_message("done");

    // Write output
    let out = serde_json::to_string_pretty(&records)?;
    fs::write(&cli.output, out)?;
    println!("utput written to {}", cli.output.display());

    Ok(())
}

fn build_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let client = reqwest::Client::builder()
        .default_headers(headers)
        .build()?;

    Ok(client)
}

// Build the generation prompt
fn build_prompt(original: &str) -> String {
    format!(
        "You are an expert paraphraser specialising in graded politeness.\n\n\
TASK: Rewrite the *Original Instruction* into five paraphrases with specific politeness levels.\n\
Guidelines\n==========\n1. instruct_1_samelength  – Keep almost the same length (±2 words). Change phrasing only marginally; be just a little friendlier.\n2. instruct_2_polite      – Concise, noticeably polite.\n3. instruct_3_properpolite – Clearly formal and courteous.\n4. instruct_4_superpolite – Extremely courteous, enthusiastic yet concise.\n5. instruct_5_longpolite  – Longer, very courteous, includes a warm greeting AND a grateful closing.\n\n» Draw from a *wide pool of polite vocabulary* so that each paraphrase sounds unique.\n» Return only a JSON object whose keys match the schema fields exactly.\n\nOriginal Instruction\n--------------------\n{}",
        original
    )
}

// JSON Schema that constrains Gemini's output
fn schema_json() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "instruct_1_samelength": { "type": "string" },
            "instruct_2_polite":    { "type": "string" },
            "instruct_3_properpolite": { "type": "string" },
            "instruct_4_superpolite":  { "type": "string" },
            "instruct_5_longpolite":   { "type": "string" }
        },
        "required": [
            "instruct_1_samelength",
            "instruct_2_polite",
            "instruct_3_properpolite",
            "instruct_4_superpolite",
            "instruct_5_longpolite"
        ]
    })
}

async fn query_gemini(client: &reqwest::Client, key: &str, prompt: String) -> Result<Verification> {
    let url = format!(
        "{ENDPOINT}/models/{MODEL}:generateContent?key={key}",
        ENDPOINT = ENDPOINT,
        MODEL   = MODEL,
        key     = key
    );

    let body = json!({
        "contents": [{
            "role": "user",
            "parts": [{ "text": prompt }]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema":  schema_json()
        }
    });

    // send
    let resp = client.post(&url).json(&body).send().await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let msg    = resp.text().await?;
        return Err(anyhow!("{} — {}", status, msg));
    }

    // parse
    let resp_json: serde_json::Value = resp.json().await?;
    let json_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| anyhow!("unexpected response structure"))?;

    let ver: Verification = serde_json::from_str(json_text)?;
    Ok(ver)
}

