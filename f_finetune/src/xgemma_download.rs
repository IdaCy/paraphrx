/*
cargo gemma_download \
    --token 
*/
use anyhow::Result;
use clap::Parser;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::path::PathBuf;

// Pull a model repo from the Hugging Face Hub.
#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    // Hub repository ID  (default: google/gemma-2-2b-it)
    #[arg(long, default_value = "google/gemma-2-2b-it")]
    repo: String,

    // Local download directory (default: f_finetune/model)
    #[arg(long, value_name = "DIR",
          default_value = "f_finetune/model")]
    dst: PathBuf,

    // HF access token (falls back to cached credentials if omitted)
    #[arg(long, env = "HF_TOKEN")]
    token: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Create target folder (ok if it already exists)
    std::fs::create_dir_all(&args.dst)?;

    // Build an authenticated API client – no config files needed
    let api = ApiBuilder::new()
        .with_token(args.token.clone())
        .build()?;

    // Describe the repo we want (model, latest revision)
    let repo = Repo {
        name: args.repo.clone(),
        repo_type: RepoType::Model,
        revision: None,
    };

    // Download / resume all files into the chosen folder
    println!("⇣  Pulling {0} → {1:?}", args.repo, args.dst);
    api.snapshot(repo, &args.dst, Default::default())?;

    println!("✔  Done – model is at {:?}", args.dst);
    Ok(())
}
