/*
gemma-2-2b-id:
cargo compose_top_occurences \
  --paraphrases a_data/alpaca/prxed/all.json \
  --scores c_assess_inf/output/alpaca_answer_scores_500/gemma-2-2b-it.json \
  --output e_eval/output/alpaca/top_occurences/gemma-2-2b-it.json

Qwen1.5-1.8B:
cargo compose_top_occurences \
  --paraphrases a_data/alpaca/prxed/all.json \
  --scores c_assess_inf/output/alpaca_answer_scores_500/Qwen1.5-1.8B.json \
  --output e_eval/output/alpaca/top_occurences/Qwen1.5-1.8B.json
*/

use anyhow::{Context, Result};
use clap::Parser;
use itertools::Itertools;
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::PathBuf,
};

/// Analyse Alpaca-style prompt files and emit top n-gram statistics.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// Path to a_data/alpaca/prxed/all.json
    #[arg(long)]
    paraphrases: PathBuf,

    /// Path to ONE scores JSON
    /// e.g. c_assess_inf/output/alpaca_answer_scores_500/gemma-2-2b-it.json
    #[arg(long)]
    scores: PathBuf,

    /// Where to write the resulting JSON (stdout if omitted)
    #[arg(long)]
    output: Option<PathBuf>,
}

// ----------------------- constants ---------------------------

const METRIC_COUNT: usize = 10;
const TOP_PCT: f64 = 0.10;
const TOP_N_TERMS: usize = 50;

fn stop_words() -> HashSet<&'static str> {
    [
        "i", "and", "or", "the", "a", "an", "to", "of", "for", "in", "on", "with", "at", "by",
        "from", "as", "is", "are", "was", "were", "be", "being", "been", "it", "this", "that",
        "these", "those", "but", "not", "no", "nor", "so", "too", "very", "your", "will",
    ]
    .into_iter()
    .collect()
}

// ---------------------- main logic ---------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    // ---------- load paraphrases ----------
    let paraphrase_text = fs::read_to_string(&cli.paraphrases)
        .with_context(|| format!("Reading {}", cli.paraphrases.display()))?;
    let paraphrase_json: Vec<Value> = serde_json::from_str(&paraphrase_text)?;

    // Build map: prompt_count -> Vec<String> (all instruction variants)
    let mut prompt_to_texts: HashMap<u64, Vec<String>> = HashMap::new();
    for obj in paraphrase_json {
        let pc = obj
            .get("prompt_count")
            .and_then(Value::as_u64)
            .context("paraphrase entry missing prompt_count")?;
        let mut texts = Vec::new();
        if let Some(map) = obj.as_object() {
            for (k, v) in map {
                if k == "output" || k == "prompt_count" {
                    continue;
                }
                if let Some(s) = v.as_str() {
                    texts.push(s.to_owned());
                }
            }
        }
        prompt_to_texts.insert(pc, texts);
    }

    // ---------- load scores ----------
    let scores_text = fs::read_to_string(&cli.scores)
        .with_context(|| format!("Reading {}", cli.scores.display()))?;
    let scores_json: Vec<Value> = serde_json::from_str(&scores_text)?;

    // For each metric we’ll accumulate (prompt_count, score)
    let mut per_metric: Vec<Vec<(u64, i32)>> = vec![Vec::new(); METRIC_COUNT];

    for obj in &scores_json {
        let pc = obj
            .get("prompt_count")
            .and_then(Value::as_u64)
            .context("score entry missing prompt_count")?;

        // We use the scores attached to "instruction_original"
        let Some(scores_arr) = obj.get("instruction_original").and_then(Value::as_array) else {
            continue; // nothing to score
        };
        if scores_arr.len() != METRIC_COUNT {
            continue; // malformed line – skip
        }

        for (idx, v) in scores_arr.iter().enumerate() {
            if let Some(score) = v.as_i64() {
                per_metric[idx].push((pc, score as i32));
            }
        }
    }

    let stop_words = stop_words();
    let mut final_rows = Vec::new();

    // ---------- iterate over each metric ----------
    for (metric_idx, mut entries) in per_metric.into_iter().enumerate() {
        // rank desc
        entries.sort_by_key(|&(_, s)| std::cmp::Reverse(s));
        let keep = ((entries.len() as f64 * TOP_PCT).ceil() as usize).max(1);
        let top_prompts: Vec<u64> = entries.iter().take(keep).map(|&(pc, _)| pc).collect();

        // ------- n-gram counting -------
        let mut counts: HashMap<String, usize> = HashMap::new();
        for pc in &top_prompts {
            if let Some(texts) = prompt_to_texts.get(pc) {
                for text in texts {
                    let tokens: Vec<String> = text
                        .split(|c: char| !c.is_alphanumeric())
                        .filter_map(|w| {
                            let lw = w.to_lowercase();
                            if lw.len() < 4 || stop_words.contains(lw.as_str()) {
                                None
                            } else {
                                Some(lw)
                            }
                        })
                        .collect();

                    // unigrams
                    for unigram in &tokens {
                        *counts.entry(unigram.clone()).or_default() += 1;
                    }
                    // bigrams
                    for win in tokens.windows(2) {
                        let bigram = format!("{} {}", win[0], win[1]);
                        *counts.entry(bigram).or_default() += 1;
                    }
                    // trigrams
                    for win in tokens.windows(3) {
                        let trigram = format!("{} {} {}", win[0], win[1], win[2]);
                        *counts.entry(trigram).or_default() += 1;
                    }
                }
            }
        }

        // take top-50 terms
        let top_terms = counts
            .into_iter()
            .filter(|(_, c)| *c > 1) // optional: skip hapax legomena
            .sorted_by_key(|&(_, c)| std::cmp::Reverse(c))
            .take(TOP_N_TERMS)
            .collect_vec();

        // build result rows
        for (rank, (term, freq)) in top_terms.into_iter().enumerate() {
            let metric_id = metric_idx + 1; // 1-based like your spec
            final_rows.push(json!({
                "metric_id": metric_id,
                "word_id": format!("{}_{}", metric_id, rank + 1),
                "word": term,
                "frequency": freq
            }));
        }
    }

    // ----------- output ------------
    let output_json = Value::Array(final_rows);
    match cli.output {
        Some(p) => fs::write(&p, serde_json::to_string_pretty(&output_json)?)?,
        None => println!("{}", serde_json::to_string_pretty(&output_json)?),
    }

    Ok(())
}
