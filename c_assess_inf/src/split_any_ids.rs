/*
cargo split_any_ids \
    f_finetune/data/all_alpaca_gemma-2-2b-it_all_ppl.json \
    100 100 100 100 100
*/

use std::{
    env,
    fs,
    io::Write,
    path::Path,
};
use clap::{Arg, Command};

fn main() -> anyhow::Result<()> {
    let matches = Command::new("split_any_ids")
        .version("1.0")
        .about("Split a JSON file by grabbing top‑level {…} objects, ignoring interior validity.")
        .arg(Arg::new("input")
            .help("Input JSON file (top-level array of objects)")
            .required(true)
            .index(1))
        .arg(Arg::new("sizes")
            .help("Max items in each split; last chunk may be smaller")
            .required(true)
            .index(2)
            .num_args(1..)
            .value_parser(clap::value_parser!(usize)))
        .get_matches();

    let input_path = Path::new(matches.get_one::<String>("input").unwrap());
    let sizes: Vec<usize> = matches
        .get_many::<usize>("sizes")
        .unwrap()
        .copied()
        .collect();

    let cwd = env::current_dir()?;
    eprintln!("🔍 cwd: {}", cwd.display());
    let raw = fs::read_to_string(input_path)
        .map_err(|e| anyhow::anyhow!("Couldn’t read `{}`: {}", input_path.display(), e))?;

    // Scan for '[' then extract every top-level {...} at bracket depth == 1
    let mut objects = Vec::new();
    let mut in_string = false;
    let mut escape = false;
    let mut bracket_depth = 0;
    let mut brace_depth = 0;
    let mut start_idx = None;

    for (i, ch) in raw.char_indices() {
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '[' => bracket_depth += 1,
            ']' => bracket_depth -= 1,
            '{' if bracket_depth == 1 => {
                if brace_depth == 0 {
                    start_idx = Some(i);
                }
                brace_depth += 1;
            }
            '{' if bracket_depth > 1 => {
                brace_depth += 1;
            }
            '}' if bracket_depth >= 1 && brace_depth > 0 => {
                brace_depth -= 1;
                if brace_depth == 0 {
                    if let Some(s) = start_idx.take() {
                        // include the '}' in the slice
                        let obj_text = &raw[s..=i];
                        objects.push(obj_text.to_string());
                    }
                }
            }
            _ => {}
        }
    }

    eprintln!("Found {} objects", objects.len());

    // prepare filenames
    let parent = input_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = input_path.file_stem().and_then(|s| s.to_str()).unwrap_or("split");
    let ext = input_path.extension().and_then(|s| s.to_str()).unwrap_or("json");

    // chunk & write
    let mut idx = 0;
    let mut start = 0;
    for &limit in &sizes {
        let end = (start + limit).min(objects.len());
        let chunk = &objects[start..end];
        let out = parent.join(format!("{stem}_part{n}.{ext}", stem=stem, n=idx+1, ext=ext));
        let mut f = fs::File::create(&out)?;
        write!(f, "[")?;
        for (i, obj) in chunk.iter().enumerate() {
            if i > 0 { write!(f, ",")?; }
            write!(f, "\n{}", obj)?;
        }
        writeln!(f, "\n]")?;
        eprintln!("  • wrote {} items to {}", chunk.len(), out.display());
        start = end;
        idx += 1;
    }

    if start < objects.len() {
        eprintln!("⚠️ {} objects left over; increase your sizes list", objects.len() - start);
    }

    Ok(())
}
