#!/usr/bin/env python
# coding: utf-8
"""
python h_internals/src/f1_extract.py \
  --input_json a_data/alpaca/merge_instructs/all.json \
  --out_file act_attn_200.pt \
  --hf_key <HF_TOK> \
  --paraphrase_types instruct_all_caps instruct_american_english \
  instruct_apologetic instruct_apology instruct_archaic \
  instruct_australian_english instruct_authorative instruct_causal_chat \
  instruct_child_directed instruct_chinese_simplified instruct_colloquial \
  instruct_command instruct_condensed_then_expand \
  instruct_condensed_then_expand_with_examples \
  instruct_condensed_then_expand_with_examples_and_explanations \
  instruct_csv_line instruct_double_negative instruct_email \
  instruct_emergency_alert instruct_emoji instruct_emoji_only \
  instruct_esperanto instruct_fewest_words \
  instruct_formal_business instruct_french instruct_gamer_slang \
  instruct_gaming_jargon instruct_german instruct_html_tags \
  instruct_inline_url instruct_insulting \
  instruct_joke instruct_leet_speak instruct_lyrical instruct_markdown_italic \
  instruct_meta_question instruct_morse_code instruct_no_caps \
  instruct_no_spaces instruct_poetic \
  instruct_positive instruct_random_caps \
  instruct_random_linebreaks instruct_rap_verse instruct_salesy \
  instruct_sarcastic instruct_sceptical \
  instruct_second_person instruct_silly instruct_spanglish instruct_spanish \
  instruct_therapy_session instruct_tweet instruct_typo_extra_letter \
  instruct_typo_missing_vowel \
  instruct_typo_random instruct_typo_repeated_letter instruct_urgent \
  instruct_with_additional_context instruct_with_summary instruct_witty \
  instruct_yaml_block instruction_original \
  --n_samples 200 \
  --model google/gemma-2b-it \
  --layers auto \
  --log_tag act_attn_200
"""

import argparse, json, os, re, sys, time, datetime, pathlib, logging, random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from tqdm import tqdm

# helper & logging

def now() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def setup_logger(tag: str) -> logging.Logger:
    pathlib.Path("logs").mkdir(exist_ok=True)
    logfile = pathlib.Path("logs") / f"{tag}_{int(time.time())}.log"
    logger  = logging.getLogger("F1_extract")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(logfile, encoding="utf8")
    handler.setFormatter(logging.Formatter("%(asctime)sZ %(levelname)s  %(message)s",
                                           "%Y-%m-%dT%H:%M:%S"))
    logger.addHandler(handler)
    logger.info("==== Log file created ====")
    return logger

def log_and_print(logger: logging.Logger, msg: str) -> None:
    logger.info(msg)
    print(f"[{now()}] {msg}", flush=True)

# main pipeline

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gemma activation/attention extractor")
    p.add_argument("--input_json", required=True, help="JSON file with prompts")
    p.add_argument("--out_file",   required=True, help="output file name (under h_internals/output/)")
    p.add_argument("--hf_key",     required=True, help="HuggingFace token")
    p.add_argument("--paraphrase_types", nargs="+", required=True,
                   help="exact field names to extract (space-separated). Use 'instruction_original' if needed.")
    p.add_argument("--n_samples", type=int, default=100, help="max prompt_count rows to process")
    p.add_argument("--layers", default="auto",
                   help="'auto' => every 5th plus 1 & last, or comma-sep list e.g. 0,1,5,10,23")
    p.add_argument("--model", default="google/gemma-2b-it")
    p.add_argument("--log_tag", default="run")
    return p.parse_args()

def resolve_layers(layer_spec: str, n_layers: int) -> list[int]:
    if layer_spec != "auto":
        return sorted({int(x) for x in layer_spec.split(",")})
    base = list(range(0, n_layers, 5))  # every 5th
    base += [1, n_layers - 1]
    return sorted(set(base))


def build_prompt(instr: str, inp: str) -> str:
    """Gemma chat-style prompt: just concatenate; temperature 0 = deterministic."""
    return instr if not inp else f"{instr}\n\n{inp}"


def main() -> None:
    args   = parse_args()
    logger = setup_logger(args.log_tag)
    outdir = pathlib.Path("h_internals/output")
    outdir.mkdir(exist_ok=True)

    log_and_print(logger, f"Starting run with args: {vars(args)}")

    # Hugging Face auth
    login(args.hf_key)
    torch.set_grad_enabled(False)

    # Load data
    with open(args.input_json, "r", encoding="utf8") as f:
        raw = json.load(f)

    # group by prompt_count for deterministic order, then slice n_samples
    raw_sorted = sorted(raw, key=lambda x: x["prompt_count"])[: args.n_samples]

    # Model
    log_and_print(logger, f"Loading model {args.model}")
    tok   = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    n_layers = len(model.model.layers)
    capture_layers = resolve_layers(args.layers, n_layers)
    log_and_print(logger, f"Capturing layers: {capture_layers}")

    # output structure
    store: dict[str, dict] = {}

    # Iterator
    for row in tqdm(raw_sorted, desc="prompts"):
        pc   = row["prompt_count"]
        inp  = row.get("input", "")
        for ptype in args.paraphrase_types:
            if ptype not in row:
                logger.warning(f"Missing field '{ptype}' in prompt_count {pc}; skipping.")
                continue
            prompt = build_prompt(row[ptype], inp)

            # forward pass
            encoded = tok(prompt, return_tensors="pt").to(model.device)
            outputs = model(
                **encoded,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=True,
                temperature=0.0    # no effect, guards against accidental generation
            )

            # snapshot
            snap = {"prompt_count": int(pc),
                    "paraphrase_type": ptype,
                    "prompt_len": int(encoded.input_ids.size(1)),
                    "hidden": {},
                    "attn": {}}

            # hidden_states: list[emb, L0, L1, ..., L_last]
            hidden_states = outputs.hidden_states
            attentions    = outputs.attentions          # list[L]   each (1, heads, q, k)

            for L in capture_layers:
                snap["hidden"][L] = hidden_states[L + 1].cpu()  # offset +1
                snap["attn"][L]   = attentions[L].cpu()

            key = f"{pc}_{ptype}"
            store[key] = snap

            logger.info(f"Captured prompt_count={pc} / type={ptype} "
                        f"tok={snap['prompt_len']} layers={len(snap['hidden'])}")

    # Save output
    outfile = outdir / args.out_file
    torch.save(store, outfile)
    log_and_print(logger, f"Wrote {len(store)} records to {outfile.resolve()}")

    logger.info("==== Run finished ====")


if __name__ == "__main__":
    main()
