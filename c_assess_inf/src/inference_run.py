#!/usr/bin/env python3
"""
Each JSON needed like:
[
  {
    "prompt_id": "uuid-string",
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

python c_assess_inf/src/inference_run.py \
python c_assess_inf/src/inference_run.py \
       a_data/alpaca/slice_100/alpaca_prx_style1_slice1.json \
       c_assess_inf/output/alpaca_prx_style1_slice1.json \
       --model google/gemma-2b-it \
       --temperature 0 \
       --max_tokens 256 \
       --type style \
       --n_samples 2





nohup python c_assess_inf/src/inference_run.py \
      a_data/alpaca/slice_100/speci_char_slice1.json \
      c_assess_inf/output/alpaca_newphras/gemma-2-2b-it/speci_char_slice1.json \
      --model google/gemma-2-2b-it \
      --batch 2048 \
      --log_every 200 \
      --type speci_char \
      > logs/speci_char_console.out 2>&1 &

nohup python c_assess_inf/src/inference_run.py \
      a_data/mmlu/prxed_moral_500/context.json \
      c_assess_inf/output/mmlu/gemma-2-2b-it/gemma-2-2b-it/context.json \
      --model google/gemma-2-2b-it \
      --batch 1024 \


nohup python c_assess_inf/src/inference_run.py \
      a_data/alpaca/slice_100/context_slice1.json \
      c_assess_inf/output/alpaca_prxed/Qwen1.5-1.8B/context_slice1.json \
      --model Qwen/Qwen1.5-1.8B \
      --batch 4096 \
      --log_every 90 \
      --type context \
      > logs/context_console.out 2>&1 &


nohup python c_assess_inf/src/inference_run.py \
      a_data/gsm8k/prxed_main_500/context.json \
      c_assess_inf/output/gsm8k/gemma-2-2b-it/context.json \
      --model Qwen/google/gemma-2-2b-it \
      --batch 2048 \
      --log_every 90 \
      --type context \
      > logs/context_console.out 2>&1 &

nohup python c_assess_inf/src/inference_run.py \
      a_data/gsm8k/prxed_main_500/language.json \
      c_assess_inf/output/gsm8k/gemma-2-2b-it/language.json \
      --model Qwen/google/gemma-2-2b-it \
      --batch 1024 \
      --log_every 90 \
      --type language \
      > logs/language_console.out 2>&1 &

nohup python c_assess_inf/src/inference_run.py \
      a_data/gsm8k/prxed_main_500/language.json \
      c_assess_inf/output/gsm8k/gemma-2-2b-it/language.json \
      --model Qwen/google/gemma-2-2b-it \
      --batch 1024 \
      --log_every 90 \
      --type language \
      > logs/language_console.out 2>&1 &

nohup python c_assess_inf/src/inference_run.py \
      a_data/gsm8k/prxed_main_500/obstruction.json \
      c_assess_inf/output/gsm8k/gemma-2-2b-it/obstruction.json \
      --model Qwen/google/gemma-2-2b-it \
      --batch 1024 \
      --log_every 90 \
      --type obstruction \
      > logs/obstruction_console.out 2>&1 &

nohup python c_assess_inf/src/inference_run.py \
      a_data/gsm8k/prxed_main_500/extra.json \
      c_assess_inf/output/gsm8k/gemma-2-2b-it/extra.json \
      --model Qwen/google/gemma-2-2b-it \
      --batch 1024 \
      --log_every 90 \
      --type context \
      > logs/context_console.out 2>&1 &



      --type extra \
      > logs/extra_console.out 2>&1 &




# tail -f run_inf_128_google-gemma-2b-it_*.log
# tail -f console.out
"""
from __future__ import annotations

import argparse
import json
import os
import logging
from datetime import datetime
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
    parser.add_argument("--log_every", type=int, default=100, help="Write a progress line every N items (default: 100)")
    parser.add_argument("--type", default="", help="String prepended to the log file name")  # ← added parameter

    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Logging – filename: run_inf_<batch>_<model-no-slash>_<YYYYmmdd_HHMMSS>.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"logs/{args.type}_run_inf_{args.batch}_{args.model.replace('/', '-')}_{timestamp}.log"  # ← prepended type
    logging.basicConfig(
        filename=log_name,
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    logging.info("==== run started ====")
    logging.info(
        "input=%s  output=%s  model=%s  batch=%s  max_tokens=%s  temp=%s",
        args.input_json,
        args.output_json,
        args.model,
        args.batch,
        args.max_tokens,
        args.temperature,
    )
    # ---------------------------------------------------------------------

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

    def batched(seq, n):
        """Simple slicer → (seq[i : i+n] for i in range(0, len(seq), n))."""
        for i in range(0, len(seq), n):
            yield seq[i : i + n]

    for idx, item in enumerate(tqdm(data, desc="generating"), 1):
        try:
            prompt_id = item.get("prompt_id", "")
            prompt_count = item.get("prompt_count")
            raw_input = item.get("input", "")

            prompt_result: Dict[str, str] = {"prompt_id": prompt_id}
            if prompt_count is not None:
                prompt_result["prompt_count"] = prompt_count

            instruction_keys = ["instruction_original"] + [
                k for k in sorted(item) if k.startswith("instruct_")
            ]
            # Build (key, prompt_text) tuples once
            variant_prompts = [
                (key, build_prompt(item[key].strip(), raw_input))
                for key in instruction_keys
            ]

            # --- NEW: truly batched generation ---------------------------------
            for batch in batched(variant_prompts, args.batch):
                batch_keys, batch_texts = zip(*batch)
                inputs = tokenizer(
                    list(batch_texts), return_tensors="pt", padding=True
                ).to(model.device)
                input_lens = inputs["attention_mask"].sum(dim=1)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                for i, key in enumerate(batch_keys):
                    start = int(input_lens[i])
                    completion_ids = outputs[i, start:]
                    prompt_result[key] = (
                        tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                    )
            # --------------------------------------------------------------------

            results.append(prompt_result)

            if idx % args.log_every == 0:
                logging.info(
                    "Processed %d / %d prompts (last prompt_id=%s)",
                    idx,
                    len(data),
                    prompt_id,
                )
        except Exception:  # noqa: BLE001
            logging.exception("FAILED on prompt_id=%s  (index %d)", prompt_id, idx)
            raise

    # Persist
    Path(args.output_json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} generations → {args.output_json}\nDone!")
    logging.info("Finished OK – wrote %d items to %s", len(results), args.output_json)


if __name__ == "__main__":
    main()
