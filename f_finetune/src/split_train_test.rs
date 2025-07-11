/*
cargo split_train_test -o f_finetune/data/output_splits

cargo split_train_test \
    -i f_finetune/data/mmlu_gemma-2-2b-it.json \
    -o f_finetune/data/output_splits_mmlu
*/

use clap::Parser;
use log::{info, warn};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use simplelog::*;
use std::collections::{HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;


#[derive(Parser, Debug)]
#[clap(name = "alpaca_splitter")]
struct Args {
    // Input JSON file path
    #[clap(short, long, default_value = "f_finetune/data/alpaca_gemma-2-2b-it.json")]
    input: PathBuf,

    // Train ratio (default 0.8)
    #[clap(short = 'r', long, default_value = "0.8")]
    train_ratio: f32,

    // Validation ratio (default 0.0)
    #[clap(short = 'v', long, default_value = "0.0")]
    val_ratio: f32,

    // Test ratio (default 0.2)
    #[clap(short = 't', long, default_value = "0.2")]
    test_ratio: f32,

    // Output directory for JSONs and logs
    #[clap(short, long, default_value = "output_splits")]
    output_dir: PathBuf,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Paraphrase {
    instruct_type: String,
    paraphrase: String,
    answer: String,
    task_score: u8,
    ranking_for_buckets: u32,
    bucket: u8,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct PromptData {
    prompt_count: u32,
    instruction_original: String,
    input: String,
    output: String,
    count_in_buckets: Vec<u32>,
    paraphrases: Vec<Paraphrase>,
}

fn main() -> anyhow::Result<()> {
    // Parse args
    let args = Args::parse();

    // Validate ratio sum approx 1.0
    let sum = args.train_ratio + args.val_ratio + args.test_ratio;
    if (sum - 1.0).abs() > 0.001 {
        eprintln!(
            "Train + Val + Test ratios must sum to 1. Got: {} + {} + {} = {}",
            args.train_ratio, args.val_ratio, args.test_ratio, sum
        );
        std::process::exit(1);
    }

    // Setup logging
    create_dir_all(&args.output_dir)?;
    create_dir_all(args.output_dir.join("logs"))?;

    CombinedLogger::init(vec![
        WriteLogger::new(
            LevelFilter::Info,
            ConfigBuilder::new().build(),
            File::create(args.output_dir.join("logs/alpaca_splitter.log"))?,
        ),
    ])?;

    info!("Starting alpaca splitter");
    info!("Input file: {:?}", args.input);
    info!(
        "Ratios - Train: {}, Val: {}, Test: {}",
        args.train_ratio, args.val_ratio, args.test_ratio
    );

    // Load JSON data
    let file = File::open(&args.input)?;
    let reader = BufReader::new(file);
    let data: Vec<PromptData> = serde_json::from_reader(reader)?;
    info!("Loaded {} prompt_count items", data.len());

    // Index by prompt_count
    let mut prompt_map = HashMap::new();
    for item in data.into_iter() {
        if prompt_map.insert(item.prompt_count, item.clone()).is_some() {
            warn!("Duplicate prompt_count found: {}", item.prompt_count);
        }
    }

    // Prepare bucket sets for bucket ranges 1..=1, 1..=2, ..., 1..=5
    // Each bucket range will contain prompt_counts which have at least one paraphrase in these buckets
    info!("Grouping prompt_counts by bucket ranges...");

    // bucket range -> set of prompt_counts
    let mut bucket_range_prompt_counts: HashMap<u8, HashSet<u32>> = HashMap::new();
    for bucket_end in 1..=5 {
        bucket_range_prompt_counts.insert(bucket_end, HashSet::new());
    }

    for (&prompt_count, prompt) in &prompt_map {
        // collect buckets of paraphrases for this prompt_count
        let buckets_present: HashSet<u8> =
            prompt.paraphrases.iter().map(|p| p.bucket).collect();

        for bucket_end in 1..=5 {
            // If any bucket in 1..=bucket_end appears, include this prompt_count
            if buckets_present.iter().any(|&b| b >= 1 && b <= bucket_end) {
                bucket_range_prompt_counts
                    .get_mut(&bucket_end)
                    .unwrap()
                    .insert(prompt_count);
            }
        }
    }

    // For each bucket range: 
    //  - get the prompt_counts
    //  - shuffle them
    //  - split into train/val/test
    //  - output train and test JSONs filtered by bucket range and bucket condition

    for bucket_end in 1..=5 {
        info!("Processing bucket range 1-{}", bucket_end);
        let mut pcs: Vec<u32> =
            bucket_range_prompt_counts.get(&bucket_end).unwrap().iter().copied().collect();

        let total = pcs.len();
        info!("Prompt_counts in bucket range 1-{}: {}", bucket_end, total);

        let mut rng = thread_rng();
        pcs.shuffle(&mut rng);

        //let train_end = (total as f32 * args.train_ratio).round() as usize;
        //let val_end = train_end + (total as f32 * args.val_ratio).round() as usize;
        let train_end = (total as f32 * args.train_ratio).floor() as usize;
        let val_end   = train_end + (total as f32 * args.val_ratio).floor() as usize;

        let train_pcs = &pcs[..train_end.min(total)];
        let val_pcs = if args.val_ratio > 0.0 && val_end <= total {
            &pcs[train_end..val_end]
        } else {
            &[]
        };
        let test_pcs = &pcs[val_end.min(total)..];

        info!(
            "Split sizes for bucket range 1-{}: train={} val={} test={}",
            bucket_end,
            train_pcs.len(),
            val_pcs.len(),
            test_pcs.len()
        );

        // Filter prompt data by these splits and bucket range:
        // For each prompt_count, filter paraphrases only up to bucket_end
        // Because bucket ranges are cumulative, keep paraphrases with bucket <= bucket_end

        let filter_prompts = |prompt_counts: &[u32]| -> Vec<PromptData> {
            prompt_counts
                .iter()
                .filter_map(|pc| prompt_map.get(pc))
                .map(|prompt| {
                    let filtered_paraphrases: Vec<Paraphrase> = prompt
                        .paraphrases
                        .iter()
                        .filter(|p| p.bucket <= bucket_end)
                        .cloned()
                        .collect();

                    PromptData {
                        prompt_count: prompt.prompt_count,
                        instruction_original: prompt.instruction_original.clone(),
                        input: prompt.input.clone(),
                        output: prompt.output.clone(),
                        count_in_buckets: prompt.count_in_buckets.clone(),
                        paraphrases: filtered_paraphrases,
                    }
                })
                .collect()
        };

        let train_data = filter_prompts(train_pcs);
        let val_data = filter_prompts(val_pcs);
        let test_data = filter_prompts(test_pcs);

        let base_name = format!("buckets_1-{}", bucket_end);

        // Write train JSON
        if !train_data.is_empty() {
            let train_path = args.output_dir.join(format!("{}_train.json", base_name));
            write_json(&train_path, &train_data)?;
            info!("Wrote train data to {:?}", train_path);
        } else {
            warn!("Train data empty for bucket range 1-{}", bucket_end);
        }

        // Write val JSON only if val_ratio > 0
        if args.val_ratio > 0.0 {
            if !val_data.is_empty() {
                let val_path = args.output_dir.join(format!("{}_val.json", base_name));
                write_json(&val_path, &val_data)?;
                info!("Wrote validation data to {:?}", val_path);
            } else {
                warn!("Validation data empty for bucket range 1-{}", bucket_end);
            }
        }

        // Write test JSON
        if !test_data.is_empty() {
            let test_path = args.output_dir.join(format!("{}_test.json", base_name));
            write_json(&test_path, &test_data)?;
            info!("Wrote test data to {:?}", test_path);
        } else {
            warn!("Test data empty for bucket range 1-{}", bucket_end);
        }
    }

    info!("All done successfully.");
    Ok(())
}

fn write_json(path: &PathBuf, data: &[PromptData]) -> anyhow::Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, data)?;
    Ok(())
}
