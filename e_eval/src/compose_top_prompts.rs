/*
gemma-2-2b-it:
cargo compose_top_prompts \
  --scores  c_assess_inf/output/alpaca_answer_scores_500/gemma-2-2b-it.json \
  --prxeds  a_data/alpaca/prxed/all.json \
  --answers c_assess_inf/output/alpaca_prxed/gemma-2-2b-it/all.json \
  --output  e_eval/output/alpaca/top_prompts/gemma-2-2b-it.json

Qwen1.5-1.8B:
cargo compose_top_prompts \
  --scores  c_assess_inf/output/alpaca_answer_scores_500/Qwen1.5-1.8B.json \
  --prxeds  a_data/alpaca/prxed/all.json \
  --answers c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/all.json \
  --output  e_eval/output/alpaca/top_prompts/Qwen1.5-1.8B.json
*/

use std::{collections::HashMap, fs};
use clap::Parser;
use serde::Serialize;
use serde_json::Value;

// CLI definition
#[derive(Parser)]
#[command(author, version, about = "Select the 10 best-scoring prompts per metric")]
struct Cli {
    // Path to model score JSON
    #[arg(long)]
    scores: String,
    // Path to paraphrased-prompt JSON
    #[arg(long)]
    prxeds: String,
    // Path to model answers JSON
    #[arg(long)]
    answers: String,
    // Output file
    #[arg(long)]
    output: String,
}

// One paraphrased prompt-answer pair together with its 10 scores
#[derive(Debug, Clone)]
struct Entry {
    prompt_count: i64,
    prx_type: String,
    scores: Vec<f64>, // len == 10
}

// Row stored in the final “top-10 per metric” file
#[derive(Serialize)]
struct OutputExample {
    prompt_count: i64,
    example_id: String,   // e.g. "3_7"
    metric_id: usize,     // 1‥10
    prx_type: String,     // paraphrase key
    scores: Vec<f64>,     // original 10-element vector
    prxed_example: String,
    answer_example: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CLI & file loading
    let cli = Cli::parse();

    let scores_raw      = fs::read_to_string(&cli.scores)?;
    let paraphrases_raw = fs::read_to_string(&cli.prxeds)?;
    let answers_raw     = fs::read_to_string(&cli.answers)?;

    let scores_json:      Vec<Value> = serde_json::from_str(&scores_raw)?;
    let paraphrases_json: Vec<Value> = serde_json::from_str(&paraphrases_raw)?;
    let answers_json:     Vec<Value> = serde_json::from_str(&answers_raw)?;

    // Build look-up maps: (prompt_count, prx_type) -> text / answer
    let mut prompt_map:  HashMap<(i64, String), String> = HashMap::new();
    let mut answer_map:  HashMap<(i64, String), String> = HashMap::new();

    for obj in &paraphrases_json {
        let pc = obj["prompt_count"].as_i64()
            .expect("prompt_count missing in paraphrases");
        for (k, v) in obj.as_object().unwrap() {
            if ["prompt_count", "output", "input"].contains(&k.as_str()) { continue; }
            if let Some(txt) = v.as_str() {
                prompt_map.insert((pc, k.clone()), txt.to_owned());
            }
        }
    }

    for obj in &answers_json {
        let pc = obj["prompt_count"].as_i64()
            .expect("prompt_count missing in answers");
        for (k, v) in obj.as_object().unwrap() {
            if k == "prompt_count" { continue; }
            if let Some(txt) = v.as_str() {
                answer_map.insert((pc, k.clone()), txt.to_owned());
            }
        }
    }

    //  Flatten the scores into entries
    let mut entries: Vec<Entry> = Vec::new();
    for obj in &scores_json {
        let pc = obj["prompt_count"].as_i64()
            .expect("prompt_count missing in scores");
        for (k, v) in obj.as_object().unwrap() {
            if ["prompt_count", "prompt_id"].contains(&k.as_str()) { continue; }
            if let Some(arr) = v.as_array() {
                if arr.len() != 10 { continue; }
                let scores: Vec<f64> = arr.iter()
                    .map(|n| n.as_f64().unwrap_or(0.0))
                    .collect();
                entries.push(Entry {
                    prompt_count: pc,
                    prx_type: k.clone(),
                    scores,
                });
            }
        }
    }

    // Average score for each paraphrase type (tie-breaker)
    let mut sums: HashMap<(usize, String), (f64, usize)> = HashMap::new();
    for e in &entries {
        for (m, &s) in e.scores.iter().enumerate() {
            let entry = sums.entry((m, e.prx_type.clone()))
                            .or_insert((0.0, 0));
            entry.0 += s;
            entry.1 += 1;
        }
    }
    let avg: HashMap<(usize, String), f64> = sums.into_iter()
        .map(|((m, p), (sum, n))| ((m, p), sum / n as f64))
        .collect();

    // Select top-10 per metric
    let mut tops: Vec<OutputExample> = Vec::new();

    for metric in 0..10 {
        // sort: 1) score desc, 2) paraphrase avg desc, 3) prompt_count asc
        let mut sorted: Vec<&Entry> = entries.iter().collect();
        sorted.sort_by(|a, b| {
            let (sa, sb) = (a.scores[metric], b.scores[metric]);
            if (sb - sa).abs() > f64::EPSILON {
                return sb.partial_cmp(&sa).unwrap();
            }
            let (ava, avb) = (
                *avg.get(&(metric, a.prx_type.clone())).unwrap(),
                *avg.get(&(metric, b.prx_type.clone())).unwrap(),
            );
            if (avb - ava).abs() > f64::EPSILON {
                return avb.partial_cmp(&ava).unwrap();
            }
            a.prompt_count.cmp(&b.prompt_count)
        });

        for (rank, entry) in sorted.into_iter().take(10).enumerate() {
            let prxed_example = prompt_map
                .get(&(entry.prompt_count, entry.prx_type.clone()))
                .cloned()
                .unwrap_or_else(|| "<prompt text not found>".into());
            let answer_example = answer_map
                .get(&(entry.prompt_count, entry.prx_type.clone()))
                .cloned()
                .unwrap_or_else(|| "<answer not found>".into());

            tops.push(OutputExample {
                prompt_count: entry.prompt_count,
                example_id:   format!("{}_{}", metric + 1, rank + 1),
                metric_id:    metric + 1,
                prx_type:     entry.prx_type.clone(),
                scores:       entry.scores.clone(),
                prxed_example,
                answer_example,
            });
        }
    }

    // Persist
    fs::write(&cli.output, serde_json::to_string_pretty(&tops)?)?;
    println!("Top-10 examples for each metric written to {}", cli.output);
    Ok(())
}
