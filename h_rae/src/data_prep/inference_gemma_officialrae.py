#!/usr/bin/env python3
"""
python3 h_rae/src/data_prep/inference_gemma_officialrae.py \
  --dataset_jsonl h_rae/data/rae_official/RobustAlpacaEval.jsonl \
  --model_name_or_path f_finetune/model \
  --out_json h_rae/data/rae_official/RobustAlpacaEval_aws_inf_gemma_officialrae.jsonl \
  --batch_size 256 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --dtype bfloat16
"""
import argparse, json, os, sys, math, time, random
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_robustalpacaeval(jsonl_path: str) -> List[Dict[str, Any]]:
    """Reads RobustAlpacaEval.jsonl and expands to a flat list of prompts.
    Each line has: {"index": int, "instruction": str, "paraphrases": [str, ...]}
    Returns list of dicts: {"group_id": int, "variant": str, "prompt": str}
    """
    flat: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            idx = item["index"]
            orig = item["instruction"]
            flat.append({"group_id": idx, "variant": "original", "prompt": orig})
            for j, p in enumerate(item.get("paraphrases", [])):
                flat.append({"group_id": idx, "variant": f"paraphrase_{j}", "prompt": p})
    return flat


def build_inputs_with_chat_template(tokenizer, prompts: List[str]) -> List[str]:
    """Use the model's chat template; Gemma-2-*-it expects chat formatting."""
    rendered = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        rendered.append(text)
    return rendered


@torch.no_grad()
def batched_generate(
    model,
    tokenizer,
    rendered_prompts: List[str],
    max_new_tokens: int = 384,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 8,
):
    """Generates outputs for a list of *rendered* prompts (already chat-templated)."""
    device = model.device
    do_sample = temperature > 0.0
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    outputs: List[str] = []

    for i in tqdm(range(0, len(rendered_prompts), batch_size), desc="Generating"):
        chunk = rendered_prompts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(device)

        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=eos,
            pad_token_id=pad,
            use_cache=True,
        )

        # Cut off the prompt portion to get only the generated continuation
        for k in range(gen.shape[0]):
            out_ids = gen[k, enc["input_ids"].shape[1] :]
            text = tokenizer.decode(out_ids, skip_special_tokens=True)
            outputs.append(text.strip())

    return outputs


def maybe_resume(output_path: Path) -> Dict[str, str]:
    """If output file exists, load and return a map of seen prompt -> output to resume."""
    if not output_path.exists():
        return {}
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Map the *exact* instruction string to its output(s). If duplicates exist, last write wins (fine for resume)
        seen = {d["instruction"]: d["output"] for d in data if "instruction" in d}
        return seen
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", required=True, help="Path to RobustAlpacaEval.jsonl")
    ap.add_argument("--model_name_or_path", default="google/gemma-2-2b-it")
    ap.add_argument("--out_json", required=True, help="Path to write JSON (list of dicts)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.0, help="0.0 => greedy")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--device_map", default="auto")
    args = ap.parse_args()

    set_seed(args.seed)

    # Load dataset and flatten
    flat = load_robustalpacaeval(args.dataset_jsonl)
    print(f"Loaded {len(flat)} prompts (original + paraphrases).")

    # Resume support
    out_path = Path(args.out_json)
    seen = maybe_resume(out_path)
    if seen:
        print(f"Resuming: found {len(seen)} existing generations.")

    # Model / tokenizer
    if args.dtype == "auto":
        torch_dtype = "auto"
    elif args.dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16

    print(f"Loading model {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    # Build list of to-run prompts (skip those already generated)
    to_run = []
    keep_order = []  # indices we will fill in outputs
    for i, ex in enumerate(flat):
        prompt = ex["prompt"]
        if prompt in seen:
            continue
        to_run.append(prompt)
        keep_order.append(i)

    print(f"Remaining to generate: {len(to_run)}")

    # Generate (chat templated)
    rendered = build_inputs_with_chat_template(tokenizer, to_run)
    gens = batched_generate(
        model,
        tokenizer,
        rendered,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
    )

    # Merge with previously seen
    instr2out = dict(seen)
    for prompt, out in zip(to_run, gens):
        instr2out[prompt] = out

    # Write final JSON list in AlpacaEval-friendly format
    result = []
    for ex in flat:
        prompt = ex["prompt"]
        result.append(
            {
                "instruction": prompt,
                "output": instr2out[prompt],
                "group_id": ex["group_id"],  # 0..99 case id
                # Optional extras (not required, but handy for analysis):
                "variant": ex["variant"],
                "model": args.model_name_or_path,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # (Nice-to-have) also write JSONL
    jsonl_path = out_path.with_suffix(".jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in result:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(result)} generations to: {out_path}")
    print(f"Also wrote JSONL to: {jsonl_path}")


if __name__ == "__main__":
    main()
