/*
cargo run \
    --manifest-path a_data/preproc/rephras/Cargo.toml \
    --release -- \
    --version-set style \
    --model "gemini-2.5-pro-preview-05-20" \
    a_data/alpaca/slice_100/alpaca_slice1.json \
    a_data/alpaca/slice_100/alpaca_prx_style1_slice1.json
*/

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{env, fs, path::PathBuf};
use tokio::time::{sleep, Duration};
use time::macros::format_description;

// logging
use chrono::Local;
use simplelog::{ConfigBuilder, LevelFilter, WriteLogger};
//use std::time::Duration;

// All the variant key-sets to support - Can contain '_' but no spaces
static VERSION_SETS: phf::Map<&'static str, &'static [&'static str]> = phf::phf_map! {
    "all" => &[
        "instruct_output_markdown",
        "instruct_one_typo_punctuation",
        "instruct_coord_to_subord",
        "instruct_future_tense",
        "instruct_polite_request",
        "instruct_dramatic",
        "instruct_sardonic",
        "instruct_joke",
        "instruct_formal_memo",
        "instruct_double_negative",
        "instruct_leet_speak"
    ],
};

// Alpaca-style record with a flexible extra map to hold any new keys
#[derive(Debug, Deserialize, Serialize)]
struct Record {
    prompt_count:        u32,
    #[serde(alias = "instruction", alias = "instruction_original")]
    instruction_original: String,

    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Parser, Debug)]
#[command(version, author, about = "Generate paraphrase variants with Gemini")]
struct Cli {
    input: PathBuf,
    output: PathBuf,

    // Which key-set to use:  style | length | obstruction | language | context
    #[arg(long, default_value = "style")]
    version_set: String,

    #[arg(long, default_value_t = 3)]
    max_attempts: u8,

    // LLM model to use
    #[arg(long, default_value = "gemini-2.5-flash-preview-05-20")]
    model: String,
}

const ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";

// How often to emit log lines during the record loop
const LOG_EVERY_N: usize = 10;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // logger setup
    let log_dir = PathBuf::from("logs");
    fs::create_dir_all(&log_dir).with_context(|| "failed to create logs directory")?;

    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let out_file_name = cli
        .output
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output.json".to_string());
    let log_path = log_dir.join(format!("{}+{}", timestamp, out_file_name));

    let log_file = fs::File::create(&log_path)
        .with_context(|| format!("failed to create log file {}", log_path.display()))?;

    WriteLogger::init(
        LevelFilter::Info,
        ConfigBuilder::new()
            .set_time_format_custom(
                format_description!("[year]-[month]-[day] [hour]:[minute]:[second]")
            )
            .build(),
        log_file,
    ).expect("failed to initialise file logger");

    log::info!("Program started");

    let keys = VERSION_SETS
        .get(cli.version_set.as_str())
        .ok_or_else(|| anyhow!("unknown version set {}", cli.version_set))?;
    let schema = schema_for(keys);

    // Read dataset
    let data = fs::read_to_string(&cli.input)
        .with_context(|| format!("failed to read {}", cli.input.display()))?;
    let mut records: Vec<Record> = serde_json::from_str(&data)?;

    log::info!("Loaded {} records from {}", records.len(), cli.input.display());

    let key = env::var("GOOGLE_API_KEY").context("GOOGLE_API_KEY not set")?;
    let client = build_client()?;

    let bar = ProgressBar::new(records.len() as u64);
    bar.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap());

    let mut processed: usize = 0;
    for rec in &mut records {
        processed += 1;
        if processed % LOG_EVERY_N == 0 {
            log::info!("Processing record {} (prompt_count {})", processed, rec.prompt_count);
        }

        let prompt = build_prompt(&rec.instruction_original, keys, &cli.version_set);
        let mut success = false;
        let mut last_error: Option<anyhow::Error> = None;


        for attempt in 1..=cli.max_attempts {
            match query_gemini(&client, &key, &schema, &prompt, &cli.model).await {
            //match query_gemini(&client, &key, &schema, prompt.clone(), &cli.model).await {
                Ok(ver) => {
                    for (k, v) in ver {
                        rec.extra.insert(k, v);
                    }
                    success = true;
                    log::info!("prompt_count {} processed successfully", rec.prompt_count);
                    break;
                }
                Err(err) => {
                    last_error = Some(err);
                    if attempt < cli.max_attempts {
                        log::warn!(
                            "prompt_count {} attempt {}/{} failed: {}",
                            rec.prompt_count,
                            attempt,
                            cli.max_attempts,
                            last_error.as_ref().unwrap()
                        );
                        sleep(Duration::from_millis(500 * u64::from(attempt))).await;
                    } else {
                        log::error!(
                            "prompt_count {} failed after {} attempts – skipping.\n\
                            Last error details:\n{}",
                            rec.prompt_count,
                            cli.max_attempts,
                            last_error.as_ref().unwrap()
                        );
                    }
                }
            }
        }

        bar.inc(1);
        if !success {
            continue;
        }
    }
    bar.finish_with_message("done");

    log::info!("All records processed – writing output to {}", cli.output.display());

    // Write output
    let out = serde_json::to_string_pretty(&records)?;
    fs::write(&cli.output, out)?;
    println!("output written to {}", cli.output.display());
    log::info!("Output written to {}", cli.output.display());

    Ok(())
}

fn build_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let client = reqwest::Client::builder()
        .default_headers(headers)
        .timeout(Duration::from_secs(90))
        .build()?;

    Ok(client)
}

// Build the generation prompt
fn build_prompt(original: &str, keys: &[&str], label: &str) -> String {
    let bullet_list = keys
        .iter()
        .map(|k| format!("* **{k}** – rewrite in the \"{label}\" variant ({k})."))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "You are an expert paraphraser.\n\
         Rewrite the *Original Instruction* in ALL of the variants listed below.\n\n\
         {bullet_list}\n\n\
         Rewrite the original instruction in the style of each key name.\n\
         Phrase every variant instruction so that its answer will be an answer to the original instruction or in some way (e.g. completeness, creativity, style, structure, efficiency, tone) better.\n\
         **Important:** Each variant must still yield an answer to the _original instruction_.\n\n\
         Return **only** one JSON object with exactly those keys.\n\n\
         Original Instruction:\n{original}"
    )
}

// JSON Schema that constrains Gemini's output
fn schema_for(keys: &[&str]) -> serde_json::Value {
    let mut props = serde_json::Map::new();
    for k in keys {
        props.insert((*k).into(), json!({ "type": "string" }));
    }
    json!({
        "type": "object",
        "properties": props,
        "required": keys,
    })
}

async fn query_gemini(
    client: &reqwest::Client,
    key: &str,
    schema: &serde_json::Value,
    prompt: &str,
    model: &str,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    let url = format!(
        "{ENDPOINT}/models/{model}:generateContent?key={key}",
        ENDPOINT = ENDPOINT,
        model  = model,
        key    = key
    );

    let body = json!({
        "contents": [{ "role": "user", "parts": [{ "text": prompt }] }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema":  schema
        }
    });

    let resp = client.post(&url).json(&body).send().await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let msg    = resp.text().await?;
        return Err(anyhow!("{} — {}", status, msg));
    }

    let resp_json: serde_json::Value = resp.json().await?;

    // Gracefully report any layout surprises with the full payload
    let json_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| {
            anyhow!(
                "unexpected response structure; full JSON from Gemini:\n{}",
                serde_json::to_string_pretty(&resp_json)
                    .unwrap_or_else(|_| "<unable to serialise>".to_string())
            )
        })?;

    // Attach the offending string if the inner deserialise blows up
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(json_text).map_err(|e| {
            anyhow!(
                "failed to parse JSON returned by Gemini: {e}\njson_text:\n{json_text}"
            )
        })?;

    Ok(map)
}
