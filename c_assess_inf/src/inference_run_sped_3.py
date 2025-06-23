#!/usr/bin/env python3
"""
Each JSON needed like:
[
  {
    "prompt_count": "uuid-string",
    "instruction_original": "Give three tips for staying healthy.",
    "input": "",  # optional – may be empty or missing
    "instruct_apologetic": "I'm sorry to ask, but could you perhaps...",
    "instruct_archaic": "Pray tell, reveal unto me...",
    ...
  },
  ...
]
"""

from __future__ import annotations

import argparse
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import signal
import atexit
import sys

import torch
from huggingface_hub import login as hf_login, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from tqdm import tqdm
import gc

# Optional imports / feature flags
try:
    from transformers import BitsAndBytesConfig  # type: ignore
    _BITSANDBYTES_OK = True
except (ImportError, AttributeError):
    BitsAndBytesConfig = None  # type: ignore
    _BITSANDBYTES_OK = False

try:
    import importlib, flash_attn                              # noqa: F401
    importlib.import_module("flash_attn.flash_attn_interface")
    _FLASH2_OK = True
except Exception:
    _FLASH2_OK = False

# Manual override: set DISABLE_FLASH_ATTN=1 at runtime to force fallback
import os as _os
if _os.getenv("DISABLE_FLASH_ATTN") == "1":
    _FLASH2_OK = False

_INFER_CTX = getattr(torch, "inference_mode", torch.no_grad)

# SPEED TWEAKS
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    for _fn in (
        "enable_flash_sdp",
        "enable_mem_efficient_sdp",
        "enable_math_sdp",
    ):
        try:
            getattr(torch.backends.cuda, _fn)(True)
        except Exception:  # covers AttributeError & other edge-cases
            pass


# Helpers
def ensure_hf_auth(token: Optional[str]) -> None:
    if token:
        hf_login(token=token, add_to_git_credential=False, new_session=True)


def assert_model_access(model_id: str, token: Optional[str]) -> None:
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
    flat_queue: list[tuple[str, str, str]] = []
    results_map: Dict[str, Dict[str, str]] = {}

    for item in data:
        prompt_count = str(item["prompt_count"])
        res_entry: Dict[str, str] = {"prompt_count": item["prompt_count"]}
        results_map[prompt_count] = res_entry

        raw_input = item.get("input", "")
        instruction_keys = ["instruction_original"] + [
            k for k in sorted(item) if k.startswith("instruct_")
        ]
        for key in instruction_keys:
            flat_queue.append(
                (prompt_count, key, build_prompt(item[key].strip(), raw_input))
            )

    return flat_queue, results_map


# Main
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-turn inference over an Alpaca paraphrase dataset."
    )
    parser.add_argument("input_json", help="Path to paraphrase dataset (.json)")
    parser.add_argument("output_json", help="Where to write model completions (.json)")
    parser.add_argument(
        "--model",
        default="google/gemma-2b-it",
        help="Model repository name on Hugging Face",
    )
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--quant",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="8-/4-bit NF4 quantisation via bitsandbytes",
    )
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--type", default="")

    args = parser.parse_args()

    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"logs/{args.type}_run_inf_{args.batch}_{args.model.replace('/', '-')}_{timestamp}.log"
    logging.basicConfig(
        filename=log_name,
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    logging.info("==== run started ====")
    logging.info(
        "input=%s  output=%s  model=%s  batch=%s  max_tokens=%s  temp=%s  quant=%s",
        args.input_json,
        args.output_json,
        args.model,
        args.batch,
        args.max_tokens,
        args.temperature,
        args.quant,
    )

    # Auth / access sanity-check
    ensure_hf_auth(args.hf_token)
    assert_model_access(args.model, args.hf_token)

    # Tokenizer
    print("Loading tokenizer & model – first run may download several GB …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, model_max_length=4096
    )

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    # Build model kwargs with graceful degradation
    if args.quant != "none" and not _BITSANDBYTES_OK:
        logging.warning("bitsandbytes not available → reverting to bf16")
        args.quant = "none"

    model_kwargs: dict = dict(device_map=args.device)
    if _FLASH2_OK:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        logging.info("Flash-Attention 2 not found → using standard attention")

    if args.quant == "none":
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        try:
            bnb_cfg = BitsAndBytesConfig(
                load_in_8bit=(args.quant == "8bit"),
                load_in_4bit=(args.quant == "4bit"),
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = bnb_cfg
        except Exception as e:  # noqa: BLE001
            logging.warning("BitsAndBytesConfig failed (%s) – falling back to bf16", e)
            args.quant = "none"
            model_kwargs["torch_dtype"] = torch.bfloat16

    # Load model (retry on flash-attn or quant failure)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    except Exception as e:  # noqa: BLE001
        if model_kwargs.pop("attn_implementation", None) == "flash_attention_2":
            logging.warning(
                "flash_attention_2 failed (%s) – retrying with standard attention",
                e,
            )
            model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        elif args.quant != "none":
            logging.warning("Quant load failed (%s) – retrying in bf16", e)
            model_kwargs = dict(
                device_map=args.device,
                torch_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
            args.quant = "none"
        else:
            raise
    model.eval()

    #  Optional torch.compile
    try:
        model = torch.compile(model)
    except Exception:  # pragma: no cover
        pass

    # Dataset
    data: List[Dict[str, str]] = json.loads(Path(args.input_json).read_text())
    if args.n_samples is not None:
        data = data[: args.n_samples]

    flat_queue, results_map = flatten_dataset(data)

    # Resume support – load existing completions
    completed_pairs = set()
    if Path(args.output_json).exists():
        try:
            existing_items = json.loads(Path(args.output_json).read_text())
            for item in existing_items:
                prompt_count = str(item["prompt_count"])
                if prompt_count in results_map:
                    results_map[prompt_count].update(item)
                for k in item:
                    if k != "prompt_count":
                        completed_pairs.add((prompt_count, k))
        except Exception as e:  # noqa: BLE001
            logging.warning("Could not load existing output (%s) – starting fresh", e)

    # Remove already-done work
    flat_queue = [
        t for t in flat_queue if (t[0], t[1]) not in completed_pairs
    ]
    flat_queue.sort(key=lambda t: len(tokenizer(t[2]).input_ids))

    # Graceful shutdown & partial save
    def _save_partial() -> None:
        Path(args.output_json).write_text(
            json.dumps(list(results_map.values()), indent=2, ensure_ascii=False)
        )

    def _handle_signal(sig_num, _frame):
        logging.info("Received signal %s – saving partial results and exiting", sig_num)
        _save_partial()
        sys.exit(0)

    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _handle_signal)
    atexit.register(_save_partial)

    # Generation loop
    for start in tqdm(range(0, len(flat_queue), args.batch), desc="generating"):
        batch_slice = flat_queue[start : start + args.batch]
        batch_ids, batch_keys, batch_texts = zip(*batch_slice)

        inputs = tokenizer(list(batch_texts), return_tensors="pt", padding=True).to(
            model.device
        )
        input_lens = inputs["attention_mask"].sum(dim=1)

        with _INFER_CTX():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        for i in range(len(batch_slice)):
            start_tok = int(input_lens[i])
            completion_ids = outputs[i, start_tok:]
            completion = tokenizer.decode(
                completion_ids, skip_special_tokens=True
            ).strip()
            results_map[batch_ids[i]][batch_keys[i]] = completion

        # Free GPU memory
        del inputs, outputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        if (start + len(batch_slice)) % args.log_every == 0:
            logging.info(
                "Processed %d / %d prompts", start + len(batch_slice), len(flat_queue)
            )

    # Persist
    Path(args.output_json).write_text(
        json.dumps(list(results_map.values()), indent=2, ensure_ascii=False)
    )
    print(f"Saved {len(results_map)} generations → {args.output_json}\nDone!")
    logging.info(
        "Finished OK – wrote %d items to %s", len(results_map), args.output_json
    )


if __name__ == "__main__":
    main()
