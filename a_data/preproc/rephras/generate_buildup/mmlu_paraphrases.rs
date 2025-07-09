/*
cargo run \
    --manifest-path a_data/preproc/rephras/Cargo.toml \
    --release -- \
    --version-set obstruction \
    b_tests/data/alpaca_10_test.json \
    b_tests/data/alpaca_10_obstruction.json
*/

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{env, fs, path::PathBuf};
use tokio::time::{sleep, Duration};

// All the variant key-sets to support - Can contain `_` but no spaces
static VERSION_SETS: phf::Map<&'static str, &'static [&'static str]> = phf::phf_map! {
    // STYLE / TONE
    "style" => &[
        "instruct_rude", "instruct_insulting",
        "instruct_neutral",
        "instruct_formal_business", "instruct_formal_academic",
        "instruct_informal", "instruct_colloquial",
        "instruct_friendly", "instruct_warm",
        "instruct_technical", "instruct_jargon",
        "instruct_legalease", "instruct_bureaucratic",
        "instruct_marketing", "instruct_salesy",
        "instruct_child_directed",
        "instruct_archaic",
        "instruct_shakespeare",
        "instruct_humorous", "instruct_playful",
        "instruct_sarcastic", "instruct_ironic",
        "instruct_poetic", "instruct_lyrical",
        "instruct_authoritative", "instruct_dogmatic",
        "instruct_apologetic", "instruct_deferential",
        "instruct_enthusiastic",
        "instruct_deadpan", "instruct_minimalist",
        "instruct_profane",
    ],

    // LENGTH / VERBOSITY─
    "length" => &[
        "one_word",
        "sentence_fragment",
        "single_sentence",
        "two_sentence",
        "short_paragraph",
        "multi_paragraph",
        "bulleted_outline",
        "numbered_steps",
        "research_paper",
        "stream_of_consciousness",
        "tldr_summary",
        "redundant_waffle",
        "nested_parentheticals",
        "recursive_self_reference",
    ],

    // OBSTRUCTIONS / NOISE (typos, case, weird chars...)
    "obstruction" => &[
        // Typos
        "typo_swap", "typo_transpose", "typo_adjacent",
        "typo_missing_vowels", "typo_repeated_letters",
        "typo_homophone", "sms_abbrev", "leet_speak",

        // Case / punctuation
        "all_caps", "no_caps", "random_caps",
        "no_punct", "extra_punct",
        "oxford_comma", "misplaced_commas",
        "em_dash_break", "parenthetical_aside",
        "interrobang", "missing_bracket", "missing_quote",

        // Special chars
        "emoji", "emoticon", "kaomoji",
        "confusable_unicode", "zero_width",
        "html_tags", "markdown_bold", "code_fence",
        "spoiler_bars", "zalgo", "gzip_b64_blob",
        "qr_ascii", "random_bytes",

        // Structure
        "random_linebreaks", "no_spaces", "reversed_text",
        "rot13", "base64", "html_comment",

        // Noise insertion
        "inline_ad", "inline_url", "hashtags", "key_smash",
    ],

    // LANGUAGE / DIALECT
    "language" => &[
        "british_english", "american_english", "australian_english",
        "singlish", "aave", "scots", "cockney",
        "hinglish", "spanglish",
        "spanish", "french", "german", "chinese_simplified",
        "klingon", "esperanto",
        "emoji_only", "morse_code",
    ],

    // CONTEXTUAL FRAMING
    "context" => &[
        "study_setup", "casual_chat", "exam_prompt",
        "formal_memo", "tech_support_ticket",
        "therapy_session", "journalist_interview",
        "roleplay_knight", "emergency_alert",
        "indirect_relay", "meta_question",
    ],
    // Register & Formality
    "register" => &[
        "very_formal", "neutral", "casual",
        "slang_heavy", "gamer_slang",
        "vulgar", "euphemistic",
        "legalese", "bureaucratic", "marketing_speak",
    ],

    // Perspective & Voice
    "voice" => &[
        "first_singular", "first_plural", "second_person", "third_person",
        "passive_voice", "impersonal_one_should",
        "past_tense", "future_tense",
    ],

    // Speech-Act
    "speech_act" => &[
        "direct_question", "indirect_question",
        "command", "polite_request", "suggestion",
        "statement", "exclamation", "apology", "greeting",
    ],

    // Question-Type
    "question_type" => &[
        "yes_no", "wh_question", "choice_question",
        "tag_question", "rhetorical_question", "nested_question",
    ],

    // Syntax Tweaks
    "syntax" => &[
        "cleft_it_is", "pseudo_cleft", "topicalization",
        "inversion", "nominalization", "coord_to_subord",
    ],

    // Polarity & Modality
    "polarity" => &[
        "positive", "negated", "double_negative", "litotes",
        "modal_must", "modal_should", "modal_may",
        "hypothetical_if", "paradox",
    ],

    // Genre / Format
    "genre" => &[
        "tweet", "sms", "email", "memo",
        "news_headline", "haiku", "rap_verse",
        "advertisement", "error_message",
        "json_format", "sql_snippet", "yaml_block", "csv_row",
        "markdown_doc", "regex_pattern",
    ],

    // Visual Layout
    "layout" => &[
        "bullet_list", "numbered_list", "table_layout",
        "checklist", "markdown_quote", "csv_line",
    ],

    // Tone & Emotion
    "tone" => &[
        "enthusiastic", "urgent", "skeptical", "confident",
        "sarcastic", "cynical", "hopeful",
        "dramatic", "melancholy",
    ],

    // Domain Jargon
    "domain" => &[
        "medical_jargon", "legal_jargon", "finance_jargon",
        "software_jargon", "physics_jargon", "gaming_jargon",
        "sports_jargon", "culinary_jargon", "fashion_jargon",
    ],

    // Cultural Reference
    "culture" => &[
        "pop_culture_meme", "historical_analogy",
        "sports_metaphor", "proverb_or_idiom",
    ],

    // Number / Symbol Swap
    "numbers" => &[
        "exact_numbers", "fuzzy_numbers", "roman_numeral",
        "scientific_notation",
    ],

    // Compression vs. Redundancy
    "compression" => &[
        "contractions", "no_contractions",
        "acronyms_spelled_out", "footnotes", "ellipsis_style",
    ],

    // Dialogue Frames
    "dialogue" => &[
        "qa_script", "timestamped_chat",
        "forum_quote", "debate_turns",
    ],

    // Self-Reflection / Edits
    "self_reflect" => &[
        "might_be_wrong", "edit_typo", "sic_marker",
    ],

    // Encoding & Ciphers
    "encoding" => &[
        "morse_code", "binary_code", "hex_code",
        "url_encoded", "caesar_cipher", "pig_latin",
        "emoji_cipher",
    ],

    // Mixed Modality
    "modality" => &[
        "see_attached_diagram", "musical_notation", "chemical_smiles",
    ],

    // Boundary Cases
    "boundary" => &[
        "empty_input", "contradictory_ask", "paradox_statement",
    ],

    // Mis-direction
    "misdirection" => &[
        "garden_path", "pun_based", "malapropism",
        "ambiguous_scope",
    ],
};

// Alpaca-style record with a flexible extra map to hold any new keys
#[derive(Debug, Deserialize, Serialize)]
struct Record {
    prompt_count:        u32,
    #[serde(alias = "instruction", alias = "prompt", alias = "question")]
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
}


const MODEL: &str = "gemini-2.5-flash-preview-04-17";
const ENDPOINT: &str = "https://generativelanguage.googleapis.com/v1beta";

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let keys = VERSION_SETS
        .get(cli.version_set.as_str())
        .ok_or_else(|| anyhow!("unknown version set {}", cli.version_set))?;
    let schema = schema_for(keys);

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
        let prompt = build_prompt(&rec.instruction_original, keys, &cli.version_set);
        let mut success = false;

        for attempt in 1..=cli.max_attempts {
            match query_gemini(&client, &key, &schema, prompt.clone()).await {
                Ok(ver) => {
                        for (k, v) in ver {
                            rec.extra.insert(k, v);
                        }
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
    prompt: String,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    let url = format!(
        "{ENDPOINT}/models/{MODEL}:generateContent?key={key}",
        ENDPOINT = ENDPOINT,
        MODEL   = MODEL,
        key     = key
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
    let json_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .ok_or_else(|| anyhow!("unexpected response structure"))?;

    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(json_text)?;
    Ok(map)
}

