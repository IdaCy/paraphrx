/*
perplexity shortcut in .cargo/config.toml !

cargo perplexity \
  --model_path google/gemma-2-2b-it \
  --data_json  a_data/alpaca/prxed/all.json \
  --out_csv    e_eval/output/alpaca_ppl.csv              \
  --n_samples  10
*/

use clap::Parser;
use csv::Writer;
use rust_bert::pipelines::common::ModelType;
use rust_bert::pipelines::generation_utils::{CausalLMModel, GenerateConfig};
use rust_bert::resources::{RemoteResource, Resource};
use rust_bert::RustBertError;
use rust_tokenizers::tokenizer::{SentencePieceTokenizer, TruncationStrategy};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tch::{kind, no_grad, Device, Kind, Tensor};

// CLI
#[derive(Parser, Debug)]
#[command(version, about = "Compute prompt-level and answer-conditional perplexity for paraphrased Alpaca datasets.")]
struct Opts {
    // HF repo ID or local directory with config.json + model.bin + tokenizer.model
    #[arg(long)]
    model_path: String,

    // Path to paraphrase JSON file
    #[arg(long)]
    data_json: PathBuf,

    // Where to write CSV results
    #[arg(long, default_value = "ppl_results.csv")]
    out_csv: PathBuf,

    // Maximum sequence length (tokens)
    #[arg(long, default_value_t = 2048)]
    max_len: usize,

    // Evaluate on GPU if built with CUDA
    #[arg(long, default_value_t = false)]
    cuda: bool,

    // Limit to first N prompt-counts (handy smoke test)
    #[arg(long)]
    n_samples: Option<usize>,
}

// JSON row (flattened)
#[derive(Debug, Deserialize)]
struct RawRecord {
    #[serde(flatten)]
    fields: HashMap<String, Value>,
}

fn main() -> Result<(), RustBertError> {
    let opts = Opts::parse();

    // Device
    let device = if opts.cuda && tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    // Model & tokenizer resources (auto-download if remote)
    let model_resource: Resource = if PathBuf::from(&opts.model_path).exists() {
        Resource::Local(opts.model_path.clone().into())
    } else {
        Resource::Remote(RemoteResource::from_pretrained(&opts.model_path))
    };

    let generate_cfg = GenerateConfig {
        model_type: ModelType::Llama, // Gemma behaves like a Llama-family model
        model_resource: model_resource.clone(),
        config_resource: model_resource.clone(),
        vocab_resource: model_resource.clone(),
        merges_resource: Resource::None, // SentencePiece -> no merges.txt
        tokenizer_resource: model_resource.clone(),
        max_length: opts.max_len as i64,
        device,
        ..Default::default()
    };

    let mut model = CausalLMModel::new(generate_cfg)?;
    model.model().to(device);
    let tokenizer: SentencePieceTokenizer = model.get_tokenizer().try_into()?;

    // Read dataset
    let raw_text = fs::read_to_string(&opts.data_json).expect("Cannot read JSON file");
    let mut records: Vec<RawRecord> = serde_json::from_str(&raw_text).expect("Bad JSON");
    if let Some(n) = opts.n_samples { records.truncate(n); }

    let mut wtr = Writer::from_path(&opts.out_csv).expect("Cannot open CSV");
    wtr.write_record(["prompt_count", "paraphrase_type", "ppl_prompt", "ppl_answer"])?;

    // Main loop
    for rec in records {
        let gold = match rec.fields.get("output") { Some(Value::String(s)) => s, _ => continue };
        let prompt_id = rec.fields.get("prompt_count").and_then(|v| v.as_i64()).unwrap_or(-1);

        for (k, v) in &rec.fields {
            if !(k.starts_with("instruct_") || k == "instruction_original") { continue; }
            let instr = match v { Value::String(s) => s, _ => continue };
            let prompt = if let Some(Value::String(inp)) = rec.fields.get("input") {
                format!("### Instruction:\n{instr}\n### Input:\n{inp}\n### Response:\n")
            } else {
                format!("### Instruction:\n{instr}\n### Response:\n")
            };

            // --- Test A: PPL(prompt) ---
            let ppl_prompt = ppl(&model, &tokenizer, &prompt, opts.max_len, None, device)?;

            // --- Test B: PPL(answer | prompt) ---
            let concat = format!("{prompt}{gold}");
            let prompt_tokens = tokenizer
                .encode(prompt.clone(), None, opts.max_len, &TruncationStrategy::LongestFirst, 0)
                .token_ids
                .len();
            let ppl_answer = ppl(&model, &tokenizer, &concat, opts.max_len, Some(prompt_tokens), device)?;

            wtr.write_record([
                prompt_id.to_string(),
                k.to_owned(),
                format!("{:.4}", ppl_prompt),
                format!("{:.4}", ppl_answer),
            ])?;
        }
    }
    wtr.flush()?;
    println!("Done -> {}", opts.out_csv.display());
    Ok(())
}

// Per-token perplexity helper
fn ppl(
    model: &CausalLMModel,
    tokenizer: &SentencePieceTokenizer,
    text: &str,
    max_len: usize,
    mask_until: Option<usize>,
    device: Device,
) -> Result<f64, RustBertError> {
    let enc = tokenizer.encode(text, None, max_len, &TruncationStrategy::LongestFirst, 0);
    let input_ids = Tensor::of_slice(&enc.token_ids).to(device).unsqueeze(0);
    let seq_len = input_ids.size()[1];

    // Labels = next-token ids + ignore-idx for last position
    let mut labels = input_ids
        .slice(1, 1, seq_len, 1)
        .cat(&Tensor::full(&[1, 1], -100, (kind::INT64_CPU, device)), 1);

    if let Some(n) = mask_until {
        if n < seq_len as usize { labels.slice(1, 0, n as i64, 1).fill_(-100); }
    }

    let logits = no_grad(|| {
        model.forward_t(
            Some(&input_ids), // input_ids
            None,            // past_key_values/cache
            None,            // attention_mask
            None,            // token_type_ids
            None,            // position_ids
            None,            // input_embeds
            None,            // encoder_outputs
            None,            // encoder_attention_mask
            false,           // train
        )
    })?;

    let log_probs = logits
        .log_softmax(-1, Kind::Float)
        .gather(-1, &labels.unsqueeze(-1), false)
        .squeeze_dim(-1)
        .neg();

    let mask = labels.ne(-100);
    let total_nll = (log_probs * mask.to_kind(Kind::Float)).sum(Kind::Float);
    let count = mask.sum(Kind::Float);
    Ok((total_nll / count).exp().double_value(&[]))
}
