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
    b_tests/data/alpaca_10_politeness.json \
    c_assess_inf/output/results.json \
        --model google/gemma-2-2b-it \
        --temperature 0 --max_tokens 256
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

# Helpers                                                                      #

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

    args = parser.parse_args()

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
    results: List[Dict[str, str]] = []

    for item in tqdm(data, desc="generating"):
        prompt_id = item.get("prompt_id", "")
        raw_input = item.get("input", "")

        variants = iter_instruction_variants(item)
        for i in range(0, len(variants), args.batch):
            chunk = variants[i : i + args.batch]
            prompts = [build_prompt(instr, raw_input) for instr in chunk]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )

            for j, instr in enumerate(chunk):
                completion = tokenizer.decode(
                    outputs[j, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                ).strip()
                results.append({"prompt_id": prompt_id, "instruction": instr, "response": completion})

    # Persist
    Path(args.output_json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} generations → {args.output_json}\nDone!")


if __name__ == "__main__":
    main()

# Example
#   HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
#   python run_inference.py paraphrases.json gemma_out.json \
#          --model google/gemma-2b-it --temperature 0.1 --max_tokens 256
#
# If you prefer a one-off CLI login instead, run:  `huggingface-cli login`.
# After that, you can omit both `HF_TOKEN` *and* `--hf_token`.
