/*
cargo gemma_download \
    --token xxx
*/

use anyhow::Result;
use clap::Parser;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    // Hub repository ID
    #[arg(long, default_value = "google/gemma-2-2b-it")]
    repo: String,

    // Destination directory
    #[arg(long, value_name = "DIR",
          default_value = "f_finetune/models/gemma-2-2b-it")]
    dst: PathBuf,

    // HF access token (falls back to cached creds)
    #[arg(long, env = "HF_TOKEN")]
    token: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.dst)?;

    // authenticated client
    let api = ApiBuilder::new().with_token(args.token.clone()).build()?;

    // “main” revision of the model repo
    let repo  = Repo::with_revision(args.repo.clone(), RepoType::Model, "main".to_string());
    let handle = api.repo(repo);

    println!("⇣  Pulling {} → {:?}", args.repo, args.dst);

    // fetch metadata → list of files (siblings)
    for sib in handle.info()?.siblings {          // RepoInfo::siblings :contentReference[oaicite:0]{index=0}
        let remote = sib.rfilename;               // Siblings::rfilename :contentReference[oaicite:1]{index=1}

        // download each file into the HF cache (or reuse it)
        let cached = handle.get(&remote)?;        // ApiRepo::get :contentReference[oaicite:2]{index=2}

        // hard-link (/ copy) into target tree
        let local = args.dst.join(&remote);
        if let Some(parent) = local.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if std::fs::hard_link(&cached, &local).is_err() {
            std::fs::copy(&cached, &local)?;
        }
    }

    println!("✔  Done - model is at {:?}", args.dst);
    Ok(())
}
