/*
gemma-2-2b-it:
cargo compose_top_occurences_across_metrics \
  --paraphrases a_data/alpaca/prxed/all.json \
  --scores c_assess_inf/output/alpaca_answer_scores_500/gemma-2-2b-it.json \
  --output e_eval/output/alpaca/top_occurences_across_metrics/gemma-2-2b-it.json

Qwen1.5-1.8B:
cargo compose_top_occurences_across_metrics \
  --paraphrases a_data/alpaca/prxed/all.json \
  --scores c_assess_inf/output/alpaca_answer_scores_500/Qwen1.5-1.8B.json \
  --output e_eval/output/alpaca/top_occurences_across_metrics/Qwen1.5-1.8B.json
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

// CLI & constants
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    #[arg(long)] paraphrases: PathBuf,
    #[arg(long)] scores: PathBuf,
    #[arg(long)] output: Option<PathBuf>,
}

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

// main
fn main() -> Result<()> {
    let cli = Cli::parse();

    // load paraphrases
    let paraphrase_text = fs::read_to_string(&cli.paraphrases)
        .with_context(|| format!("reading {}", cli.paraphrases.display()))?;
    let paraphrase_json: Vec<Value> = serde_json::from_str(&paraphrase_text)?;

    // prompt_count -> Vec<all instruction strings>
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

    // load & aggregate scores
    let scores_text = fs::read_to_string(&cli.scores)
        .with_context(|| format!("reading {}", cli.scores.display()))?;
    let scores_json: Vec<Value> = serde_json::from_str(&scores_text)?;

    // Vec<(prompt_count, total_score)>
    let mut all_scores: Vec<(u64, i32)> = Vec::new();

    for obj in &scores_json {
        let pc = obj
            .get("prompt_count")
            .and_then(Value::as_u64)
            .context("score entry missing prompt_count")?;

        let Some(scores_arr) = obj.get("instruction_original").and_then(Value::as_array) else {
            continue;
        };
        if scores_arr.len() != METRIC_COUNT {
            continue;
        }

        let total: i32 = scores_arr
            .iter()
            .filter_map(|v| v.as_i64())
            .map(|x| x as i32)
            .sum();

        all_scores.push((pc, total));
    }

    // pick the global top-10 %
    all_scores.sort_by_key(|&(_, s)| std::cmp::Reverse(s));
    let keep = ((all_scores.len() as f64 * TOP_PCT).ceil() as usize).max(1);
    let top_prompts: Vec<u64> = all_scores.iter().take(keep).map(|&(pc, _)| pc).collect();

    // n-gram counting
    let stop_words = stop_words();
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

    // top 50 n-grams
    let top_terms = counts
        .into_iter()
        .filter(|(_, c)| *c > 1)
        .sorted_by_key(|&(_, c)| std::cmp::Reverse(c))
        .take(TOP_N_TERMS);

    // metric_id 0 denotes “across metrics”
    let mut output_rows = Vec::new();
    for (rank, (term, freq)) in top_terms.enumerate() {
        output_rows.push(json!({
            "metric_id": 0,
            "word_id": format!("0_{}", rank + 1),
            "word": term,
            "frequency": freq
        }));
    }

    // write
    let json_out = Value::Array(output_rows);
    match cli.output {
        Some(p) => fs::write(&p, serde_json::to_string_pretty(&json_out)?)?,
        None => println!("{}", serde_json::to_string_pretty(&json_out)?),
    }

    Ok(())
}
