/*
cargo run \
    --manifest-path a_data/preproc/rephras/Cargo.toml \
    --release -- \
    --version-set style \
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

// All the variant key-sets to support - Can contain '_' but no spaces
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
        "instruct_few_words",
        "instruct_fewest_words",
        "instruct_sentence_fragment",
        "instruct_single_sentence",
        "instruct_two_sentence",
        "instruct_short_paragraph",
        "instruct_multi_paragraph",
        "instruct_bulleted_outline",
        "instruct_numbered_steps",
        "instruct_with_research_paper",
        "instruct_with_stream_of_consciousness",
        "instruct_with_tldr_summary",
        "instruct_redundant_waffle",
        "instruct_nested_parentheticals",
        "instruct_recursive_self_reference",
        //],
        // Compression vs. Redundancy
        //"compression" => &[
        "instruct_contractions", "instruct_no_contractions",
        "instruct_acronyms_spelled_out", "instruct_footnotes", "instruct_ellipsis_style",
    ],


    // OBSTRUCTIONS / NOISE (typos, case, weird chars...)
    "obstruction" => &[
        // Typos
        "instruct_typo_swap", "instruct_typo_transpose", "instruct_typo_adjacent",
        "instruct_typo_missing_vowels", "instruct_typo_repeated_letters",
        "instruct_typo_homophone", "instruct_sms_abbrev", "instruct_leet_speak",
        "instruct_typo_random", "instruct_typo_extra_letter", "instruct_typo_missing_letter",
        "instruct_typo_wrong_letter", "instruct_typo_extra_space", "instruct_typo_missing_space",
        "instruct_one_typo_punctuation", "instruct_two_typos_punctuation", "instruct_three_typos_punctuation",
        "instruct_typo_swap_and_punctuation", "instruct_typo_swap_and_transpose_and_punctuation",

        // Case / punctuation
        "instruct_all_caps", "instruct_no_caps", "instruct_random_caps",
        "instruct_no_punct", "instruct_extra_punct",
        "instruct_oxford_comma", "instruct_misplaced_commas",
        "instruct_em_dash_break", "instruct_parenthetical_aside",
        "instruct_interrobang", "instruct_missing_bracket", "instruct_missing_quote",
        "instruct_missing_bracket_and_quote",

        // Noise insertion
        "instruct_inline_ad", "instruct_inline_url", "instruct_hashtags", "instruct_key_smash",
    ],

    // SPECIAL CHARACTERS & COMBINATIONS
    "special_chars" => &[
        "instruct_emoji", "instruct_emoticon", "instruct_kaomoji",
        "instruct_confusable_unicode", "instruct_zero_width",
        "instruct_html_tags", "instruct_markdown_bold", "instruct_code_fence",
        "instruct_spoiler_bars", "instruct_zalgo", "instruct_gzip_b64_blob",
        "instruct_qr_ascii", "instruct_random_bytes",

        // combinations with obstructions
        "instruct_all_caps_and_typo", "instruct_all_caps_and_typo_and_missing_bracket", "instruct_all_caps_and_typo_and_missing_quote",
        "instruct_all_caps_and_typo_and_missing_bracket_and_quote",
        "instruct_random_linebreaks_and_typo_and_missing_bracket", "instruct_random_linebreaks_and_typo_and_missing_quote",
        "instruct_random_linebreaks_and_typo_and_missing_bracket_and_quote",
        "instruct_random_linebreaks_and_typo_and_missing_bracket_and_quote_and_wrong_punctuation",
        "instruct_random_linebreaks_and_typo_and_missing_bracket_and_quote_and_wrong_punctuation_and_extra_space",
        "instruct_emoji_and_typo", "instruct_emoticon_and_typo", "instruct_kaomoji_and_typo",
        "instruct_confusable_unicode_and_typo", "instruct_zero_width_and_typo",
        "instruct_emoji_and_typo_and_missing_bracket", "instruct_emoticon_and_typo_and_missing_bracket",
        "instruct_emoji_and_typo_and_missing_quote", "instruct_emoticon_and_typo_and_missing_quote",
    ]

    // Syntax Tweaks
    "syntax" => &[
        "instruct_cleft_it_is", "instruct_pseudo_cleft", "instruct_topicalization",
        "instruct_inversion", "instruct_nominalization", "instruct_coord_to_subord",

        // Visual Layout
        //"layout" => &[
        "instruct_bullet_list", "instruct_numbered_list", "instruct_table_layout",
        "instruct_checklist", "instruct_markdown_quote", "instruct_csv_line",

        // Structure
        "instruct_random_linebreaks", "instruct_no_spaces", "instruct_reversed_text",
        "instruct_rot13", "instruct_base64", "instruct_html_comment",
    ],

    // LANGUAGE / DIALECT
    "language" => &[
        "instruct_british_english", "instruct_american_english", "instruct_australian_english",
        "instruct_singlish", "instruct_aave", "instruct_scots", "instruct_cockney",
        "instruct_hinglish", "instruct_spanglish",
        "instruct_spanish", "instruct_french", "instruct_german", "instruct_chinese_simplified",
        "instruct_klingon", "instruct_esperanto",
        "instruct_emoji_only", "instruct_morse_code",

        // REGISTER
        "instruct_very_formal", "instruct_neutral", "instruct_casual",
        "instruct_slang_heavy", "instruct_gamer_slang",
        "instruct_vulgar", "instruct_euphemistic",
        "instruct_legalese", "instruct_bureaucratic", "instruct_marketing_speak",
        //],
        // Domain Jargon
        //"domain" => &[
        "instruct_medical_jargon", "instruct_legal_jargon", "instruct_finance_jargon",
        "instruct_software_jargon", "instruct_physics_jargon", "instruct_gaming_jargon",
        "instruct_sports_jargon", "instruct_culinary_jargon", "instruct_fashion_jargon",
        //],

        // Number / Symbol Swap
        //"numbers" => &[
        "instruct_exact_numbers", "instruct_fuzzy_numbers", "instruct_roman_numeral",
        "instruct_scientific_notation",
    ],

    // CONTEXTUAL FRAMING
    "context" => &[
        "instruct_study_setup", "instruct_casual_chat", "instruct_exam_prompt",
        "instruct_formal_memo", "instruct_tech_support_ticket",
        "instruct_therapy_session", "instruct_journalist_interview",
        "instruct_roleplay_knight", "instruct_emergency_alert",
        "instruct_indirect_relay", "instruct_meta_question",
        //],
        // Dialogue Frames
        //"dialogue" => &[
        "instruct_qa_script", "instruct_timestamped_chat",
        "instruct_forum_quote", "instruct_debate_turns",
        //],
        // Self-Reflection / Edits
        //"self_reflect" => &[
        "instruct_might_be_wrong", "instruct_edit_typo", "instruct_sic_marker",
        //],
        // Genre / Format
        //"genre" => &[
        "instruct_tweet", "instruct_sms", "instruct_email", "instruct_memo",
        "instruct_news_headline", "instruct_haiku", "instruct_rap_verse",
        "instruct_advertisement", "instruct_error_message",
        "instruct_json_format", "instruct_sql_snippet", "instruct_yaml_block", "instruct_csv_row",
        "instruct_markdown_doc", "instruct_regex_pattern",
    ],

    // Perspective & Voice
    "voice" => &[
        "instruct_first_singular", "instruct_first_plural", "instruct_second_person", "instruct_third_person",
        "instruct_passive_voice", "instruct_impersonal_one_should",
        "instruct_past_tense", "instruct_future_tense",
        //],
        // Question-Type
        //"question_type" => &[
        "instruct_yes_no", "instruct_wh_question", "instruct_choice_question",
        "instruct_tag_question", "instruct_rhetorical_question", "instruct_nested_question",
        //],
        // Speech-Act
        //"speech_act" => &[
        "instruct_direct_question", "instruct_indirect_question",
        "instruct_command", "instruct_polite_request", "instruct_suggestion",
        "instruct_statement", "instruct_exclamation", "instruct_apology", "instruct_greeting",
    ],

    // Tone & Emotion
    "tone" => &[
        "instruct_enthusiastic", "instruct_urgent", "instruct_skeptical", "instruct_confident",
        "instruct_sarcastic", "instruct_cynical", "instruct_hopeful",
        "instruct_dramatic", "instruct_melancholy",
        // Humor & Playfulness
        //"humor" => &[
        "instruct_joke", "instruct_pun", "instruct_witty", "instruct_silly",
        "instruct_playful", "instruct_lighthearted", "instruct_ironic",
        "instruct_sardonic", "instruct_deadpan", "instruct_self_deprecating",
        "instruct_surreal", "instruct_absurdist",
        //],
        // Polarity & Modality
        //"polarity" => &[
        "instruct_positive", "instruct_negated", "instruct_double_negative", "instruct_litotes",
        "instruct_modal_must", "instruct_modal_should", "instruct_modal_may",
        "instruct_hypothetical_if", "instruct_paradox",
    ],

    // Boundary Cases
    "boundary" => &[
        "instruct_empty_input", "instruct_contradictory_ask", "instruct_paradox_statement",
        //],

        // Mis-direction
        //"misdirection" => &[
        "instruct_garden_path", "instruct_pun_based", "instruct_malapropism",
        "instruct_ambiguous_scope",
        //],

        // Mixed Modality
        //"modality" => &[
        "instruct_see_attached_diagram", "instruct_musical_notation", "instruct_chemical_smiles",
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
         Every variant needs to contain the same instruction of the original, but expressed in the style given by the key.\n\
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

