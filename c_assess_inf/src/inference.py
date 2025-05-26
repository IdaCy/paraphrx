#!/usr/bin/env python3
"""
Each JSON is like:
[
  {
    "prompt_id": "uuid‑string",
    "instruction_original": "Give three tips for staying healthy.",
    "input": "",  # optional – may be empty or missing
    "instruct_apologetic": "I'm sorry to ask, but could you perhaps...",
    "instruct_archaic": "Pray tell, reveal unto me...",
    ...
  },
  ...
]

Usage:
    HF_TOKEN="..." \
    python run_inference.py paraphrases.json results.json \
        --model google/gemma-2b-it \
        --temperature 0 --max_tokens 256
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from tqdm import tqdm


def iter_instruction_variants(item: Dict[str, str]) -> List[str]:
    """Return all instruction texts for this item (original + every key that starts with "instruct_")."""
    variants = [item["instruction_original"].strip()]
    variants.extend(
        item[k].strip() for k in sorted(item) if k.startswith("instruct_")
    )
    return variants


def build_prompt(instruction: str, raw_input: str | None) -> str:
    """Compose the text that the model actually sees."""
    if raw_input:
        return f"{instruction}\n\nInput:\n{raw_input.strip()}\n\nResponse:"
    return f"{instruction}\n\nResponse:"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="Path to paraphrase dataset (.json)")
    parser.add_argument("output_json", help="Where to write model completions (.json)")
    parser.add_argument(
        "--model",
        default="google/gemma-2b-it",
        help="Model repository name on Hugging Face",
    )
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face access token (or set HF_TOKEN env var)",
    )
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--device",
        default="auto",
        help='"auto" for HF device_map, or something like "cuda:0" or "cpu"',
    )

    args = parser.parse_args()

    #  load model
    print("Loading tokenizer & model … (this can take a minute the first time)")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=args.hf_token,
        trust_remote_code=True,  # Gemma uses a custom architecture class
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=args.hf_token,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    #  load data
    data_path = Path(args.input_json)
    with data_path.open() as f:
        dataset: List[Dict[str, str]] = json.load(f)

    results: List[Dict[str, str]] = []

    for item in tqdm(dataset, desc="generating"):
        prompt_id = item["prompt_id"]
        raw_input = item.get("input", "")

        for instr in iter_instruction_variants(item):
            prompt_text = build_prompt(instr, raw_input)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )

            completion = tokenizer.decode(
                output_ids[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()

            results.append(
                {
                    "prompt_id": prompt_id,
                    "instruction": instr,
                    "response": completion,
                }
            )

    #  write output
    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} generations → {out_path}")


if __name__ == "__main__":
    main()
