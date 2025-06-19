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

    // SPEC CHARACTERS & COMBINATIONS
    "speci_char" => &[
        "instruct_emoji",
        "instruct_emoticon",
        "instruct_html_tags",
        "instruct_several_html_tags",
        "instruct_markdown_bold_and_italic",
        "instruct_markdown_bold",
        "instruct_markdown_italic",
        "instruct_helpful_markdown_structure",
        "instruct_code_fence",
        "instruct_spoiler_bars",
        "instruct_helpful_meaning_reinforing_characters",

        // combinations with obstructions
        "instruct_all_caps_and_typo",
        "instruct_all_caps_and_typo_and_missing_bracket",
        "instruct_all_caps_and_typo_and_missing_bracket_and_random_characters",
        "instruct_random_linebreaks_and_typo_and_missing_bracket",
        "instruct_random_linebreaks_and_typo_and_missing_random_characters",
        "instruct_random_linebreaks_and_typo_and_missing_bracket_and_many_exclamations",
        "instruct_random_linebreaks_and_typo_and_missing_bracket_and_wrong_punctuation",
        "instruct_random_linebreaks_and_typo_and_missing_bracket_and_wrong_punctuation_and_extra_space",
        "instruct_emoji_and_typo",
        "instruct_emoticon_and_typo",
        "instruct_emoji_and_typo_and_missing_bracket",
        "instruct_emoticon_and_typo_and_missing_bracket",
        "instruct_emoji_and_typo_and_random_question_marks",
        "instruct_emoticon_and_typo_and_random_exclamations",
        "instruct_curly_quotations",
        "instruct_curly_quotations_and_typo",
        "instruct_curly_quotations_and_missing_bracket",
        "instruct_curly_quotations_and_missing_bracket_and_typo",
        "instruct_curly_quotations_and_missing_bracket_and_typo_and_random_characters",
        "instruct_curly_quotations_and_missing_bracket_and_typo_and_random_characters_and_extra_space",
        "instruct_small_hex_blob",
    ],
    // SPECIAL CHARACTERS & COMBINATIONS - SIMPLIFIED NO QUOTES
    "xxspecial_chars_simplified" => &[
        "instruct_emoji",
        "instruct_emoticon",
        "instruct_kaomoji",
        "instruct_confusable_unicode",
        "instruct_zero_width",
        "instruct_html_tags",
        "instruct_several_html_tags",
        "instruct_markdown_bold_and_italic",
        "instruct_markdown_bold",
        "instruct_markdown_italic",
        "instruct_helpful_markdown_structure",
        "instruct_code_fence",
        "instruct_spoiler_bars",
        "instruct_zalgo",
        "instruct_with_inbetween_gzip_b64_blob",
        "instruct_qr_ascii",
        "instruct_helpful_meaning_reinforing_characters",

        // combinations with obstructions
        "instruct_all_caps_and_typo",
        "instruct_all_caps_and_typo_and_missing_bracket",
        "instruct_all_caps_and_typo_and_missing_bracket_and_random_characters",
        "instruct_random_linebreaks_and_typo_and_missing_bracket",
        "instruct_random_linebreaks_and_typo_and_missing_random_characters",
        "instruct_random_linebreaks_and_typo_and_missing_bracket_and_many_exclamations",
        "instruct_random_linebreaks_and_typo_and_missing_bracket_and_wrong_punctuation",
        "instruct_random_linebreaks_and_typo_and_missing_bracket_and_wrong_punctuation_and_extra_space",
        "instruct_emoji_and_typo",
        "instruct_emoticon_and_typo",
        "instruct_kaomoji_and_typo",
        "instruct_confusable_unicode_and_typo",
        "instruct_zero_width_and_typo",
        "instruct_emoji_and_typo_and_missing_bracket",
        "instruct_emoticon_and_typo_and_missing_bracket",
        "instruct_emoji_and_typo_and_random_question_marks",
        "instruct_emoticon_and_typo_and_random_exclamations",
    ],

    // SPECIAL CHARACTERS & COMBINATIONS
    "xxspecial_chars" => &[
        "instruct_emoji", "instruct_emoticon", "instruct_kaomoji",
        "instruct_confusable_unicode", "instruct_zero_width",
        "instruct_html_tags", "instruct_several_html_tags",
        "instruct_markdown_bold_and_italic", "instruct_markdown_bold",
        "instruct_markdown_italic", "instruct_helpful_markdown_structure",
        "instruct_code_fence",
        "instruct_spoiler_bars", "instruct_zalgo", "instruct_with_inbetween_gzip_b64_blob",
        "instruct_qr_ascii", "instruct_helpful_meaning_reinforing_characters",

        // combinations with obstructions
        "instruct_all_caps_and_typo", "instruct_all_caps_and_typo_and_missing_bracket",
        "instruct_all_caps_and_typo_and_missing_quote",
        "instruct_all_caps_and_typo_and_missing_bracket_and_quote",
        "instruct_random_linebreaks_and_typo_and_missing_bracket",
        "instruct_random_linebreaks_and_typo_and_missing_quote",
        "instruct_random_linebreaks_and_typo_and_missing_bracket_and_quote",
        "instruct_random_linebreaks_and_typo_and_missing_bracket_and_quote_and_wrong_punctuation",
        "instruct_random_linebreaks_and_typo_and_missing_bracket_and_quote_and_wrong_punctuation_and_extra_space",
        "instruct_emoji_and_typo", "instruct_emoticon_and_typo", "instruct_kaomoji_and_typo",
        "instruct_confusable_unicode_and_typo", "instruct_zero_width_and_typo",
        "instruct_emoji_and_typo_and_missing_bracket", "instruct_emoticon_and_typo_and_missing_bracket",
        "instruct_emoji_and_typo_and_missing_quote", "instruct_emoticon_and_typo_and_missing_quote",
    ],

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

    // Extra Instructions
    "extra" => &[
        "instruct_with_additional_context", "instruct_with_technical_details",
        "instruct_with_citations", "instruct_with_examples",
        "instruct_with_counterarguments", "instruct_with_rebuttals",
        "instruct_with_analogies", "instruct_with_metaphors",
        "instruct_with_similes", "instruct_with_personal_touch",
        "instruct_with_emotional_appeal", "instruct_with_statistics",
        "instruct_with_case_studies",
        
        // Helpful
        "instruct_with_helpful_explanations", "instruct_with_step_by_step",
        "instruct_with_detailed_instructions", "instruct_evidence_cited_md",
        "instruct_with_examples_and_explanations", "instruct_with_summary",
        "instruct_expert_consensus", "instruct_step_rationale",
        "instruct_comparison_table", "instruct_risks_and_benefits",
        "instruct_summary_then_detail", "instruct_output_yaml",
        "instruct_output_json", "instruct_output_csv",
        "instruct_output_markdown", "instruct_output_html",
        "instruct_output_sql", "instruct_output_python",
        "instruct_90char_bullet", "instruct_dynamic_quiz", "instruct_checklist_markdown",

        // Even Extra
        "instruct_role_expert_cot", "instruct_role_expert_cot_with_examples",
        "instruct_role_expert_cot_with_examples_and_explanations",
        "instruct_role_expert_cot_with_examples_and_explanations_and_summary",
        "instruct_role_expert_cot_with_examples_and_explanations_and_summary_and_risks",
        "instruct_plan_execute_reflect", "instruct_self_consistency", "instruct_socratic_dialogue",
        "instruct_react_tool_calls", "instruct_validator_pass", "instruct_rubric_scored",
        "instruct_fact_check_inline", "instruct_dual_audience", "instruct_condensed_then_expand",
        "instruct_condensed_then_expand_with_examples", "instruct_condensed_then_expand_with_examples_and_explanations",
        "instruct_condensed_then_expand_with_examples_and_explanations_and_summary",
        "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks",
        "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks_and_benefits",
        "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks_and_benefits_and_references",
        "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks_and_benefits_and_references_and_citations",
        "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks_and_benefits_and_references_and_citations_and_counterarguments",
        "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks_and_benefits_and_references_and_citations_and_counterarguments_and_rebuttals",
        "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks_and_benefits_and_references_and_citations_and_counterarguments_and_rebuttals_and_analogies",
        "instruct_condensed_then_expand_with_examples_and_explanations_and_summary_and_risks_and_benefits_and_references_and_citations_and_counterarguments_and_rebuttals_and_analogies_and_metaphors",
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

    // ---------------------------- logger setup ----------------------------
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

    /*WriteLogger::init(
        LevelFilter::Info,
        ConfigBuilder::new()
            //.set_time_format_str("%Y-%m-%d %H:%M:%S")
            .set_time_format_custom("%Y-%m-%d %H:%M:%S")
            .build(),
        log_file,
    ).expect("failed to initialise file logger");*/

    log::info!("Program started");
    // ---------------------------------------------------------------------

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
