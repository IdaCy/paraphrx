/*
cargo results_find_issues \
    --in-dir <input_dir> \
    --out-dir <output_dir> \
    --last-prompt-count 500

cargo results_find_issues \
    --in-dir c_assess_inf/output/gsm8k/gemma-2-9b-it/scores_500 \
    --out-dir c_assess_inf/output/gsm8k/gemma-2-9b-it/scores_issues_500 \
    --last-prompt-count 500
*/

use std::{
    collections::HashSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use chrono::Local;
use clap::Parser;
use log::{error, info, LevelFilter};
use simplelog::{Config, WriteLogger};

/// Scan JSON files for missing prompt_count values.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// Directory that contains the input JSON files
    #[arg(long = "in-dir", value_name = "PATH")]
    in_dir: PathBuf,

    /// Directory where *_issues.json files will be written
    #[arg(long = "out-dir", value_name = "PATH")]
    out_dir: PathBuf,

    /// Highest prompt_count that is expected to exist (default: 500)
    #[arg(long = "last-prompt-count", default_value_t = 500)]
    last_prompt_count: u32,
}

fn main() {
    // ----- parse CLI --------------------------------------------------------
    let cli = Cli::parse();

    // ----- initialise logging ----------------------------------------------
    if let Err(e) = fs::create_dir_all("logs") {
        eprintln!("Cannot create log directory: {e}");
        std::process::exit(1);
    }
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
    let log_path = format!("logs/{timestamp}.log");

    WriteLogger::init(
        LevelFilter::Info,
        Config::default(),
        fs::File::create(&log_path).expect("Cannot open log file"),
    )
    .expect("Failed to initialise logger");

    info!("Started - in_dir: {:?}, out_dir: {:?}", cli.in_dir, cli.out_dir);

    // Create out_dir if needed
    if let Err(e) = fs::create_dir_all(&cli.out_dir) {
        error!("Cannot create out_dir {:?}: {e}", cli.out_dir);
        std::process::exit(1);
    }

    // main loop over *.json files
    let Ok(entries) = fs::read_dir(&cli.in_dir) else {
        error!("Cannot read input directory {:?}", cli.in_dir);
        std::process::exit(1);
    };

    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();

        // skip non-JSON files
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }

        // skip files we generated earlier ( *_issues.json ) ─
        if path
            .file_name()
            .and_then(|s| s.to_str())
            .map_or(false, |name| name.ends_with("_issues.json"))
        {
            continue;
        }

        // process the genuine data file
        match process_file(&path, &cli.out_dir, cli.last_prompt_count) {
            Ok(true)  => info!("{:?}: issues file written", path.file_name().unwrap()),
            Ok(false) => info!("{:?}: no missing prompt_count", path.file_name().unwrap()),
            Err(e)    => error!("{:?}: {e}", path.file_name().unwrap()),
        }
    }


    info!("Finished ✅");
}

/// Returns Ok(true) if an *_issues.json file was created.
fn process_file(path: &Path, out_dir: &Path, last_prompt_count: u32) -> anyhow::Result<bool> {
    let data = fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&data)?;
    let arr = json.as_array().ok_or_else(|| anyhow::anyhow!("Top-level JSON is not an array"))?;

    let mut present: HashSet<u32> = HashSet::new();
    for obj in arr {
        if let Some(v) = obj.get("prompt_count") {
            if let Some(n) = v.as_u64()
                .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
            {
                present.insert(n as u32);
            }
        }
    }

    let mut missing: Vec<u32> = (1..=last_prompt_count)
        .filter(|n| !present.contains(n))
        .collect();

    if missing.is_empty() {
        return Ok(false);
    }

    // Create issues vector in requested string format
    let issues: Vec<String> = missing.drain(..)
        .map(|n| format!("id {n}: missing"))
        .collect();

    // Build output filename: <stem>_issues.json
    let mut out_path = out_dir.join(path.file_stem().unwrap());
    out_path.set_file_name(format!(
        "{}_issues",
        out_path.file_name().unwrap().to_string_lossy()
    ));
    out_path.set_extension("json");

    let json_text = serde_json::to_string_pretty(&issues)?;
    fs::File::create(&out_path)?.write_all(json_text.as_bytes())?;

    Ok(true)
}