#!/usr/bin/env python3
"""
Each JSON needed like:
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

Running:
    export HF_TOKEN="..."

and then:

python c_assess_inf/src/run_inference.py \
       a_data/alpaca/slice_100/alpaca_prx_style1_slice1.json \
       c_assess_inf/output/alpaca_prx_style1_slice1.json \
       --model google/gemma-2b-it \
       --temperature 0 \
       --max_tokens 256 \
       --n_samples 2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from huggingface_hub import login as hf_login, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from tqdm import tqdm

# Helpers

def ensure_hf_auth(token: Optional[str]) -> None:
    """Authenticate with the Hub *once* for the current process."""
    if token:
        hf_login(token=token, add_to_git_credential=False, new_session=True)


def assert_model_access(model_id: str, token: Optional[str]) -> None:
    """Raise a clear, early error if the user’s token cannot access the model."""
    try:
        HfApi().model_info(model_id, token=token)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Token doesn’t have access to `{model_id}`."
        ) from e


def iter_instruction_variants(item: Dict[str, str]) -> List[str]:
    variants = [item["instruction_original"].strip()]
    variants.extend(item[k].strip() for k in sorted(item) if k.startswith("instruct_"))
    return variants


def build_prompt(instruction: str, raw_input: str | None) -> str:
    if raw_input:
        return f"{instruction}\n\nInput:\n{raw_input.strip()}\n\nResponse:"
    return f"{instruction}\n\nResponse:"

# Main                                                                         #

def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-turn inference over an Alpaca paraphrase dataset.")
    parser.add_argument("input_json", help="Path to paraphrase dataset (.json)")
    parser.add_argument("output_json", help="Where to write model completions (.json)")
    parser.add_argument(
        "--model",
        default="google/gemma-2b-it",
        help="Model repository name on Hugging Face (default: google/gemma-2b-it)",
    )
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face access token (defaults to $HF_TOKEN env var)",
    )
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens to generate (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    parser.add_argument("--device", default="auto", help="Device map – 'auto', 'cuda:0', 'cpu', … (default: auto)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for prompt variants (default: 1)")
    parser.add_argument("--n_samples", type=int, default=None, help="Limit the number of dataset items processed (default: all)")

    args = parser.parse_args()

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!!!!!!!!! TEMPORARY HACK !!!!!!!!!! OVERRIDING TO 2B MODEL !!!!!!!!!!
    if args.model != "google/gemma-2-2b-it":
        print(
            f"Requested model '{args.model}' ignored; "
            "using 'google/gemma-2-2b-it' instead."
        )
    args.model = "google/gemma-2-2b-it"

    orig_out = Path(args.output_json)
    forced_dir = Path("c_assess_inf/output/alpaca/gemma-2-2b-it")
    forced_dir.mkdir(parents=True, exist_ok=True)

    forced_path = forced_dir / orig_out.name    # keep their filename

    if forced_path != orig_out:
        print(
            f"Requested output '{orig_out}' ignored; "
            f"saving to '{forced_path}' instead."
        )
    args.output_json = str(forced_path)
    # !!!!!!!!!! TEMPORARY HACK END !!!!!!!!!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Authenticate and sanity-check access
    ensure_hf_auth(args.hf_token)
    assert_model_access(args.model, args.hf_token)

    # Load model + tokenizer
    print("Loading tokenizer & model – first run may download several GB …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    # Stream through dataset -> generate completions
    data: List[Dict[str, str]] = json.loads(Path(args.input_json).read_text())
    if args.n_samples is not None:
        data = data[: args.n_samples]
    results: List[Dict[str, str]] = []

    for item in tqdm(data, desc="generating"):
        prompt_id = item.get("prompt_id", "")
        prompt_count = item.get("prompt_count")
        raw_input = item.get("input", "")

        prompt_result: Dict[str, str] = {"prompt_id": prompt_id}
        if prompt_count is not None:
            prompt_result["prompt_count"] = prompt_count

        # Collect key / instruction text pairs so we can store outputs under the same key names
        instruction_keys = ["instruction_original"] + [k for k in sorted(item) if k.startswith("instruct_")]

        for key in instruction_keys:
            instr = item[key].strip()
            prompt_text = build_prompt(instr, raw_input)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )

            completion = tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()

            prompt_result[key] = completion

        results.append(prompt_result)

    # Persist
    Path(args.output_json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} generations → {args.output_json}\nDone!")


if __name__ == "__main__":
    main()
