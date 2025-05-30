/*
cargo run --manifest-path c_assess_inf/Cargo.toml \
    --release -- \
    --version-set politeness \
    c_assess_inf/output/results_all_eval.json
*/

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use phf::phf_map;
use serde::Deserialize;
use serde_json::{Map as JsonMap, Value};
use std::{collections::HashMap, fs, path::PathBuf};

// paraphrase key-sets
static VERSION_SETS: phf::Map<&'static str, &'static [&'static str]> = phf_map! {
    "politeness" => &[
        "instruction_original",
        "instruct_1_samelength",
        "instruct_2_polite",
        "instruct_3_properpolite",
        "instruct_4_superpolite",
        "instruct_5_longpolite"
    ]
};

// data models
#[derive(Debug, Deserialize)]
struct EvalObj {
    is_correct: bool,
    score_0_to_5: i64,
    // we ignore the explanation here
}

#[derive(Debug, Deserialize)]
struct Record {
    prompt_count: u32,

    // the model answers (and eval objects) live here
    #[serde(flatten)]
    extra: JsonMap<String, Value>,
}

// CLI
#[derive(Parser, Debug)]
#[command(version, author, about = "Summarise paraphrase evaluation scores")]
struct Cli {
    #[arg(long, default_value = "style")]
    version_set: String,

    /// JSON produced by assess_results
    eval_file: PathBuf,
}

// helpers
fn mean(data: &[f64]) -> f64 {
    data.iter().copied().sum::<f64>() / data.len() as f64
}

fn std_dev(data: &[f64]) -> f64 {
    let m = mean(data);
    (data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64).sqrt()
}

fn median(data: &mut [f64]) -> f64 {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = data.len() / 2;
    if data.len() % 2 == 0 {
        (data[mid - 1] + data[mid]) / 2.0
    } else {
        data[mid]
    }
}

// main
fn main() -> Result<()> {
    let cli = Cli::parse();
    let keys = VERSION_SETS
        .get(cli.version_set.as_str())
        .ok_or_else(|| anyhow!("unknown version set {}", cli.version_set))?;

    // load JSON
    let raw = fs::read_to_string(&cli.eval_file)
        .with_context(|| format!("failed to read {}", cli.eval_file.display()))?;
    let records: Vec<Record> = serde_json::from_str(&raw)?;

    // aggregates (per key)
    let mut per_key_scores: HashMap<&str, Vec<f64>> = HashMap::new();
    let mut per_key_correct: HashMap<&str, Vec<bool>> = HashMap::new();

    // per-question discrepancy stats
    let mut per_question_delta: Vec<f64> = Vec::new();
    let mut per_question_stdev: Vec<f64> = Vec::new();

    for rec in &records {
        let mut scores_this_rec: Vec<f64> = Vec::new();

        for &k in *keys {
            let eval_key = format!("{k}_eval");
            let Some(eval_val) = rec.extra.get(&eval_key) else {
                return Err(anyhow!(
                    "record {} missing {eval_key}",
                    rec.prompt_count
                ));
            };
            let obj: EvalObj = serde_json::from_value(eval_val.clone())?;

            per_key_scores.entry(k).or_default().push(obj.score_0_to_5 as f64);
            per_key_correct.entry(k).or_default().push(obj.is_correct);

            scores_this_rec.push(obj.score_0_to_5 as f64);
        }

        // intra-question stats
        if !scores_this_rec.is_empty() {
            let min = scores_this_rec.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = scores_this_rec.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            per_question_delta.push(max - min);
            per_question_stdev.push(std_dev(&scores_this_rec));
        }
    }

    // reporting
    println!("\n=== Summary for version-set \"{}\" ({} records) ===",
             cli.version_set, records.len());

    println!("\nPer-paraphrase key:");
    println!("{:<30} {:>6} {:>6} {:>6} {:>8}",
             "key", "mean", "std", "med", "acc%");
    for &k in *keys {
        let mut scores = per_key_scores.remove(k).unwrap_or_default();
        let mean_v = mean(&scores);
        let std_v  = std_dev(&scores);
        let med_v  = median(&mut scores);
        let acc = per_key_correct
            .remove(k).unwrap_or_default()
            .iter()
            .filter(|&&b| b)
            .count() as f64
            / records.len() as f64 * 100.0;

        println!("{:<30} {:>6.2} {:>6.2} {:>6.2} {:>7.1}",
                 k, mean_v, std_v, med_v, acc);
    }

    println!("\nPer-question discrepancy (across all paraphrases):");
    println!("  mean range  : {:5.2}",  mean(&per_question_delta));
    println!("  mean stdev  : {:5.2}",  mean(&per_question_stdev));
    println!("  max  range  : {:5.2}",  per_question_delta.iter().cloned().fold(0./0., f64::max));
    println!("  max  stdev  : {:5.2}",  per_question_stdev .iter().cloned().fold(0./0., f64::max));

    println!("\nDone.");
    Ok(())
}
