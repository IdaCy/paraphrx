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

chmod +x c_assess_inf/src/inference_run_betterbatch.py
run_log=logs/$(basename "$0")_$(date +%Y%m%d_%H%M%S).out
chmod +x run_inference.sh
pgrep -af inference_run_betterbatch.py 


./run_inference.sh \
  a_data/alpaca/slice_100/speci_char_slice1.json \
  c_assess_inf/output/alpaca_newphras/gemma-2-2b-it/speci_char_slice1.json \
  --model google/gemma-2-2b-it \
  --batch 256 \
  --type speci_char &

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
import gc


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


def build_prompt(instruction: str, raw_input: str | None) -> str:
    if raw_input:
        return f"{instruction}\n\nInput:\n{raw_input.strip()}\n\nResponse:"
    return f"{instruction}\n\nResponse:"


# Batching helpers

def flatten_dataset(
    data: List[Dict[str, str]]
) -> tuple[list[tuple[str, str, str]], Dict[str, Dict[str, str]]]:
    """
    Return
      • flat_queue – list of (prompt_id, key, prompt_text) tuples
      • results_map – dict mapping prompt_id -> result-dict ready for completions
    """
    flat_queue: list[tuple[str, str, str]] = []
    results_map: Dict[str, Dict[str, str]] = {}

    for item in data:
        prompt_id = item.get("prompt_id", "")
        # Prepare per-item result shell
        res_entry: Dict[str, str] = {"prompt_id": prompt_id}
        if "prompt_count" in item:
            res_entry["prompt_count"] = item["prompt_count"]
        results_map[prompt_id] = res_entry

        raw_input = item.get("input", "")
        instruction_keys = ["instruction_original"] + [
            k for k in sorted(item) if k.startswith("instruct_")
        ]
        for key in instruction_keys:
            flat_queue.append(
                (prompt_id, key, build_prompt(item[key].strip(), raw_input))
            )

    return flat_queue, results_map

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

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    # Stream through dataset → real batching across *all* items
    data: List[Dict[str, str]] = json.loads(Path(args.input_json).read_text())
    if args.n_samples is not None:
        data = data[: args.n_samples]

    flat_queue, results_map = flatten_dataset(data)

    for start in tqdm(range(0, len(flat_queue), args.batch), desc="generating"):
        batch_slice = flat_queue[start : start + args.batch]
        batch_ids, batch_keys, batch_texts = zip(*batch_slice)

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

        # Scatter completions back to their owning prompt-dict
        for i in range(len(batch_slice)):
            start_tok = int(input_lens[i])
            completion_ids = outputs[i, start_tok:]
            completion = tokenizer.decode(
                completion_ids, skip_special_tokens=True
            ).strip()
            results_map[batch_ids[i]][batch_keys[i]] = completion

        # FREE GPU MEMORY AFTER THIS BATCH
        del inputs, outputs                    # drop Tensor references
        torch.cuda.empty_cache()               # return cached blocks
        torch.cuda.ipc_collect()               # release CUDA IPC handles
        gc.collect()                           # Python GC

        if (start + len(batch_slice)) % args.log_every == 0:
            logging.info(
                "Processed %d / %d prompts",
                start + len(batch_slice),
                len(flat_queue),
            )

    # Collect final list in original order
    results: List[Dict[str, str]] = list(results_map.values())

    # Persist
    Path(args.output_json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} generations → {args.output_json}\nDone!")
    logging.info("Finished OK – wrote %d items to %s", len(results), args.output_json)


if __name__ == "__main__":
    main()
