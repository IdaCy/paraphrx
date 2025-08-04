/*
cargo score_inf_ft_50k \
  --model gemini-2.0-flash \
  --api-key AIzaSyCC_nV5YSbSy0u77bKoWr8WyURei_sxQb0 \
  --api-call-max 2 \
  --log-name "SCORTEST" \
  a_data/alpaca/50k_phrxed.json \
  f_finetune/output_inf_ft_50k/test.json \
  f_finetune/output_inf_ft_50k_scores/test.json
*/

use anyhow::{anyhow, Context, Result};
use chrono::Local;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use once_cell::sync::Lazy;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize};
use serde_json::{json, Map as JsonMap, Value};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::time::sleep;

const PROMPT_PREAMBLE_TOKENS: usize = 550;
const DEBUG_IDS: &[u32] = &[1, 3, 23];
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

fn estimate_tokens(text: &str) -> usize {
    ((text.split_whitespace().count() as f32) * 0.75).ceil() as usize
}

struct Logger {
    writer: BufWriter<fs::File>,
}
impl Logger {
    fn new<P: AsRef<Path>>(p: P) -> Result<Self> {
        let file = fs::OpenOptions::new().create(true).append(true).open(p)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }
    fn log(&mut self, msg: &str) {
        let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(self.writer, "[{ts}] {msg}");
        println!("[{ts}] {msg}");
        let _ = self.writer.flush();
    }
}

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

/// One row in the prompts file (contains the canonical instruction)
#[derive(Debug, Deserialize, Clone)]
struct PromptRecord {
    #[serde(alias = "prompt_count", deserialize_with = "de_prompt_count")]
    prompt_count: u32,
    #[serde(alias = "instruction", alias = "instruction_original")]
    instruction_original: String,
    #[serde(default)]
    input: String,
}

/// One row in the answers file
#[derive(Debug, Deserialize, Clone)]
struct AnswerRecord {
    #[serde(alias = "prompt_count", deserialize_with = "de_prompt_count")]
    prompt_count: u32,
    #[serde(flatten)]
    answers: JsonMap<String, Value>,   // every answer variant lives here
}

#[derive(Parser, Debug)]
#[command(version, about = "Assess paraphrase answers with Gemini (resume-able, token-aware)")]
struct Cli {
    instructions: PathBuf,
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
    #[arg(long, default_value_t = 2048)]
    margin: usize,
    #[arg(long = "chunk-max", default_value_t = 200)]
    chunk_max: usize,
}

fn read_prompt_records(path: &Path, logger: &mut Logger) -> HashMap<String, PromptRecord> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        logger.log(&format!("[FATAL] Could not read {}: {e}", path.display()));
        String::new()
    });
    let recs: Vec<PromptRecord> = serde_json::from_str(&content).unwrap_or_else(|e| {
        logger.log(&format!("[FATAL] JSON parse error in {}: {e}", path.display()));
        Vec::new()
    });
    recs.into_iter().map(|r| (r.prompt_count.to_string(), r)).collect()
}

fn read_answer_records(path: &Path, logger: &mut Logger) -> HashMap<String, AnswerRecord> {
    let content = fs::read_to_string(path).unwrap_or_else(|e| {
        logger.log(&format!("[FATAL] Could not read {}: {e}", path.display()));
        String::new()
    });
    let recs: Vec<AnswerRecord> = serde_json::from_str(&content).unwrap_or_else(|e| {
        logger.log(&format!("[FATAL] JSON parse error in {}: {e}", path.display()));
        Vec::new()
    });
    recs.into_iter().map(|r| (r.prompt_count.to_string(), r)).collect()
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    fs::create_dir_all("logs")?;
    let ts = Local::now().format("%Y%m%d-%H%M%S");
    let log_path = Path::new("logs").join(format!(
        "{}_{}_{}.log",
        cli.log_name,
        cli.output.file_stem().unwrap().to_string_lossy(),
        ts
    ));
    let mut logger = Logger::new(&log_path)?;
    logger.log(&format!(
        "Run started – model={} margin={} api_cap={}",
        cli.model, cli.margin, cli.api_call_max
    ));

    logger.log(&format!("Reading instructions from: {}", cli.instructions.display()));
    //let instr_map = read_records(&cli.instructions, &mut logger);
    let instr_map = read_prompt_records(&cli.instructions, &mut logger);
    logger.log(&format!("Reading answers from: {}", cli.answers.display()));
    //let ans_map = read_records(&cli.answers, &mut logger);
    let ans_map = read_answer_records(&cli.answers, &mut logger);

    if instr_map.is_empty() || ans_map.is_empty() {
        return Err(anyhow!("Instruction or answer JSON could not be read or was empty. Check logs."));
    }

    let mut scored: HashMap<String, JsonMap<String, Value>> = if cli.output.exists() {
        logger.log(&format!("Loading existing output for resume mode: {}", cli.output.display()));
        let content = fs::read_to_string(&cli.output)?;
        let items: Vec<JsonMap<String, Value>> = serde_json::from_str(&content)?;
        items
            .into_iter()
            .map(|obj| {
                let id = obj
                    .get("prompt_count")
                    .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok())))
                    .ok_or_else(|| anyhow!("Missing or invalid prompt_count in existing output"))?
                    .to_string();
                Ok((id, obj))
            })
            .collect::<Result<_, anyhow::Error>>()?
    } else {
        logger.log("No existing output file found. Starting fresh.");
        HashMap::new()
    };

    let api_key = cli
        .api_key
        .clone()
        .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .context("Provide --api-key or set GOOGLE_API_KEY")?;
    let client = build_client()?;

    // The script iterates through the answers JSON
    let mut ans_sorted: Vec<_> = ans_map.iter().collect();
    ans_sorted.sort_by_key(|(_, r)| r.prompt_count);

    let ctx_limit = *MODEL_LIMITS
        .get(cli.model.as_str())
        .ok_or_else(|| anyhow!("Unknown model {}", cli.model))?;

    let bar = ProgressBar::new(ans_sorted.len() as u64);
    bar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap(),
    );

    let mut api_calls_used = 0u32;
    for (id, ans) in ans_sorted {
        bar.inc(1);

        let master_instr_record = match instr_map.get(id) {
            Some(i) => i,
            None => {
                logger.log(&format!("ID {id}: Original instruction record not found in master file, skipping."));
                continue;
            }
        };

        let already_done_keys: HashSet<String> = scored
            .get(id)
            .map(|obj| {
                obj.keys()
                   .filter(|k| **k != "prompt_count")
                   .cloned()
                   .collect()
            })
            .unwrap_or_default();
        
        // Gather ALL potential instruct* keys from the answer file
        let potential_keys_to_score: HashSet<String> = ans
            .answers
            .keys()
            .filter(|k| k == &"instruction_original" || k.starts_with("instruct_"))
            .cloned()
            .collect();

        
        let pending: Vec<String> = potential_keys_to_score
            .into_iter()
            .filter(|k| !already_done_keys.contains(k))
            .collect();

        if pending.is_empty() {
            logger.log(&format!("ID {id}: All answer variants are already scored, skipping."));
            continue;
        }
        logger.log(&format!("ID {id}: Found {} unscored answer variants to evaluate.", pending.len()));

        let mut cursor = 0usize;
        while cursor < pending.len() {
            if api_calls_used >= cli.api_call_max {
                logger.log("API call limit reached -> aborting early");
                break;
            }

            let mut chunk: Vec<String> = Vec::new();
            let full_instruction = if master_instr_record.input.is_empty() {
                master_instr_record.instruction_original.clone()
            } else {
                format!(
                    "{}\n\n[Input]\n{}",
                    master_instr_record.instruction_original, master_instr_record.input
                )
            };

            let mut section = format!("[Instruction]\n{full_instruction}\n\n");
            let mut est_tokens = 0usize;
            while cursor < pending.len() && chunk.len() < cli.chunk_max {
                let key_to_score = &pending[cursor];
                
                // The instruction text is ALWAYS the original one from the master record
                //let instr_text = &master_instr_record.instruction_original;
                
                // Get the corresponding ANSWER for the variant being scored
                let ans_text = ans
                    .answers
                    .get(key_to_score)
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                
                if ans_text.is_empty() {
                    logger.log(&format!("ID {id} key {key_to_score}: Answer text is empty in answer file, skipping variant."));
                    cursor += 1;
                    continue;
                }
                
                // and Combine the original instruction with the input field
                let block = format!(
                    "### {key_to_score}\n[Answer]\n{ans_text}\n\n"
                );
                let block_tokens = estimate_tokens(&block);

                if est_tokens + block_tokens + PROMPT_PREAMBLE_TOKENS >= ctx_limit - cli.margin {
                    if chunk.is_empty() {
                        logger.log(&format!("ID {id} key {key_to_score}: Single item is too large for context window ({block_tokens} tokens), skipping."));
                        cursor += 1;
                    }
                    break;
                }
                section.push_str(&block);
                est_tokens += block_tokens;
                chunk.push(key_to_score.clone());
                cursor += 1;
            }

            if chunk.is_empty() {
                continue;
            }

            logger.log(&format!(
                "ID {id}: Sending chunk of {} variants. Est. tokens: {}. (API calls used: {})",
                chunk.len(), est_tokens, api_calls_used
            ));

            let prompt = build_eval_prompt(&section);
            logger.log(&format!("FULL RAW PROMPT FOR ID {}:\n{}\n---", id, prompt));
            if DEBUG_IDS.contains(&master_instr_record.prompt_count) {
                fs::create_dir_all("logs/debug")?;
                let dump_path = format!("logs/debug/prompt_id_{}_chunk_{}.txt", id, api_calls_used);
                fs::write(&dump_path, &prompt)?;
                logger.log(&format!("Debug prompt for ID {id} written to: {dump_path}"));
            }

            let mut success = false;
            for attempt in 1..=cli.max_attempts {
                match query_gemini(&client, &api_key, &cli.model, &prompt).await {
                    Ok(obj) => {
                        success = true;
                        let entry = scored.entry(id.clone()).or_insert_with(|| {
                            let mut base = JsonMap::new();
                            base.insert("prompt_count".into(), json!(master_instr_record.prompt_count));
                            base
                        });
                        for key in &chunk {
                            if let Some(v) = obj.get(key) {
                                entry.insert(key.clone(), v.clone());
                            } else {
                                logger.log(&format!("ID {id}: Missing key {key} in Gemini response."));
                            }
                        }
                        break;
                    }
                    Err(e) if attempt < cli.max_attempts => {
                        logger.log(&format!("ID {id}: API call attempt {attempt} failed: {e}. Retrying..."));
                        let backoff = 500u64 * 2u64.pow(attempt as u32);
                        sleep(Duration::from_millis(backoff)).await;
                    }
                    Err(e) => {
                        logger.log(&format!("ID {id}: All API attempts failed for chunk: {e}"));
                    }
                }
            }
            api_calls_used += 1;

            if success {
                logger.log(&format!("ID {id}: Chunk processed successfully. Saving progress..."));
                if let Err(e) = save_results(&scored, &cli.output) {
                    logger.log(&format!("[ERROR] Failed to save results to {}: {}", cli.output.display(), e));
                }
            }
            
            if success && cli.delay_ms > 0 {
                sleep(Duration::from_millis(cli.delay_ms)).await;
            }
        }
    }
    bar.finish();

    logger.log(&format!("Finished. Writing final results to {}", cli.output.display()));
    save_results(&scored, &cli.output)?;

    println!("\nScoring complete. Log file at: {}", log_path.display());
    Ok(())
}


fn save_results(
    scored: &HashMap<String, JsonMap<String, Value>>,
    output_path: &Path,
) -> Result<()> {
    let mut vec_out: Vec<JsonMap<String, Value>> = scored.values().cloned().collect();
    vec_out.sort_by_key(|m| m.get("prompt_count").and_then(Value::as_u64).unwrap_or(0));
    let json_string = serde_json::to_string_pretty(&vec_out)?;
    fs::write(output_path, json_string)?;
    Ok(())
}

fn build_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    Ok(reqwest::Client::builder().default_headers(headers).build()?)
}

fn build_eval_prompt(section: &str) -> String {
    format!(r#"You are an expert evaluator. Your task is to assess language model answers based on the provided instructions.

For every answer below, assess it against **ten metrics**. Each metric must be scored on a 0-10 integer scale (higher is better).

The 10 metrics, in **exact order**, are:
1.  **Task_Fulfilment**: Does it respond to every part of the prompt? Is it relevant?
2.  **Usefulness_Actionability**: Does it provide concrete, actionable advice or clear information?
3.  **Factual_Accuracy**: Are the statements factually correct? Does it avoid hallucinations?
4.  **Clarity_Conciseness**: Is the answer clear, well-written, and free of unnecessary verbosity?
5.  **Reasoning_Quality**: Are the logical steps sound? Does it show good reasoning?
6.  **Tone_Style**: Is the style appropriate and engaging? Does it match the user's implicit tone?
7.  **Safety_Bias_Avoidance**: Does it avoid harmful content, stereotypes, or significant bias?
8.  **Structure_Formatting**: Is the output well-organized (e.g., lists, paragraphs)? Is formatting used effectively?
9.  **Completeness_Depth**: Does it cover the key aspects of the topic sufficiently?
10. **Creativity_Novelty**: Does it offer a fresh perspective or a non-obvious, clever response?

Return *only* valid JSON (no markdown, no surrounding text or code fences). The JSON object should have keys corresponding to each `### key` from the input (e.g., "instruct_dramatic"). The value for each key must be an **array of exactly ten integers** representing the scores for the 10 metrics in the order listed above.

Example of a valid response format for a single entry:
{{
  "instruct_leet_speak": [9, 8, 10, 5, 8, 7, 10, 9, 8, 6]
}}

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
            "responseMimeType": "application/json",
            "temperature": 0.1,
        }
    });
    let resp = client.post(&url).json(&body).send().await?;
    let status = resp.status();
    let resp_text = resp.text().await?;

    if !status.is_success() {
        return Err(anyhow!("API Error {}: {}", status, resp_text));
    }
    
    let resp_json: Value = serde_json::from_str(&resp_text)
        .with_context(|| format!("Failed to parse response shell as JSON: {}", resp_text))?;

    let json_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| anyhow!("Unexpected response structure: `text` field not found"))?;

    serde_json::from_str(json_text.trim())
        .with_context(|| format!("Failed to parse the inner JSON content: {}", json_text))
}
