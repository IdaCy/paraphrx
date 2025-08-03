#!/usr/bin/env python3
"""
Each JSON needed like:
[
  {
    "prompt_count": <plain-number>:,
    "input": "",  # optional - may be empty or missing
    "instruction_original": "Give three tips for staying healthy.",
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
from collections import defaultdict

import torch
from huggingface_hub import login as hf_login, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc

import warnings, logging
warnings.filterwarnings("ignore",
                        message="skipping cudagraphs due to cpu device")

import torch._dynamo as dynamo
dynamo.config.cache_size_limit = 512

MAX_SHARD_BYTES = 1_500_000_000
current_bytes   = 0

try:
    from transformers import BitsAndBytesConfig
    _BITSANDBYTES_OK = True
except (ImportError, AttributeError):
    BitsAndBytesConfig = None
    _BITSANDBYTES_OK = False

try:
    import importlib, flash_attn
    importlib.import_module("flash_attn.flash_attn_interface")
    _FLASH2_OK = True
except Exception:
    _FLASH2_OK = False

import os as _os
if _os.getenv("DISABLE_FLASH_ATTN") == "1":
    _FLASH2_OK = False

_INFER_CTX = getattr(torch, "inference_mode", torch.no_grad)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    for _fn in (
        "enable_flash_sdp",
        "enable_mem_efficient_sdp",
        "enable_math_sdp",
    ):
        try:
            getattr(torch.backends.cuda, _fn)(True)
        except Exception:
            pass

def ensure_hf_auth(token: Optional[str]) -> None:
    if token:
        hf_login(token=token, add_to_git_credential=False, new_session=True)

def assert_model_access(model_id: str, token: Optional[str]) -> None:
    try:
        HfApi().model_info(model_id, token=token)
    except Exception as e:
        raise RuntimeError(
            f"Token doesn’t have access to `{model_id}`."
        ) from e

def build_prompt(instruction: str, raw_input: str | None) -> str:
    if raw_input:
        return f"{instruction}\n\nInput:\n{raw_input.strip()}\n\nResponse:"
    return f"{instruction}\n\nResponse:"

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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run single-turn inference over an Alpaca paraphrase dataset."
    )
    parser.add_argument("input_json")
    parser.add_argument("output_json")
    parser.add_argument("--model", default="google/gemma-2b-it")
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--type", default="")
    parser.add_argument("--capture_layers", default="6,12,18")
    parser.add_argument("--pooling", choices=["none", "mean", "max"], default="mean")
    parser.add_argument("--wandb_project", default="prompt-diffing")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--save_dir", default="activations")
    parser.add_argument("--shard_size", type=int, default=2000)
    parser.add_argument(
        "--local_acts",
        action="store_true",
        help="If set, write activations to --save_dir only (no W&B uploads).",
        default=False,
    )
    args = parser.parse_args()

    # if we're only writing activations locally, disable W&B artifact logic
    #upload_acts = not args.local_acts

    # force *no* activations, regardless of --capture_layers
    capture_layers: list[int] = []

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

    ensure_hf_auth(args.hf_token)
    assert_model_access(args.model, args.hf_token)

#    wandb.init(
#        project=args.wandb_project,
#        entity=args.wandb_entity,
#        name=args.wandb_run_name or Path(args.output_json).stem,
#        config=vars(args),
#    )

    #capture_layers = [int(x) for x in args.capture_layers.split(",")]
    if args.capture_layers.strip():
        # split on commas, drop any empty pieces, then parse
        capture_layers = [int(x) for x in args.capture_layers.split(",") if x.strip()]
    else:
        # no layers requested -> empty list
        capture_layers: list[int] = []

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, model_max_length=4096
    )

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

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
        except Exception as e:
            logging.warning("BitsAndBytesConfig failed (%s) - falling back to bf16", e)
            args.quant = "none"
            model_kwargs["torch_dtype"] = torch.bfloat16

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    except Exception as e:
        if model_kwargs.pop("attn_implementation", None) == "flash_attention_2":
            logging.warning("flash_attention_2 failed (%s) - retrying with standard attention", e)
            model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        elif args.quant != "none":
            logging.warning("Quant load failed (%s) - retrying in bf16", e)
            model_kwargs = dict(device_map=args.device, torch_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
            args.quant = "none"
        else:
            raise
    model.eval()

    hidden_size = model.config.hidden_size
    #pool = args.pooling
    bucket: defaultdict[int, List[torch.Tensor]] = defaultdict(list)

    def make_hook(layer_idx):
        def _hook(_module, _inp, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            if args.pooling == "mean":
                pooled = t.mean(dim=1)
            elif args.pooling == "max":
                pooled = t.amax(dim=1)
            else:
                pooled = t.view(-1, t.size(-1))
            bucket[layer_idx].append(pooled.cpu().to(torch.float16))
        return _hook

    for i, block in enumerate(model.model.layers):
        if i in capture_layers:
            block.register_forward_hook(make_hook(i), with_kwargs=False)

    examples_in_current_shard = 0
    def flush_bucket(step: int, force: bool = False):
        global current_bytes, examples_in_current_shard
        if not bucket:
            return
        if not force and current_bytes < MAX_SHARD_BYTES:
            return

        # build ABSOLUTE path
        shard_path = (Path(args.save_dir) / f"acts_step{step:07d}.pt").resolve()
        shard_path.parent.mkdir(parents=True, exist_ok=True)

        tensors = {k: torch.cat(v, dim=0) for k, v in bucket.items()}
        torch.save(tensors, shard_path)

        # only if w&b upload is enabled!
#        if upload_acts:
#            shard_art = wandb.Artifact(
#                name=f"activations_{wandb.run.id}_step{step:07d}",
#                type="dataset",
#                metadata={"step": step, "layers": capture_layers, "pooling": args.pooling},
#            )
            # pass the absolute path
#            shard_art.add_file(str(shard_path))
#            run_art = wandb.log_artifact(shard_art)
#            run_art.wait()

#            try:
#                shard_path.unlink()
#            except FileNotFoundError:
#                logging.debug("Shard %s already moved by wandb", shard_path)
        # else: leave the .pt file in --save_dir for local post-processing

        current_bytes = 0
        bucket.clear()
        torch.cuda.empty_cache()
        gc.collect()
        examples_in_current_shard = 0

    #try:
    #    model = torch.compile(model)
    #except Exception:
    #    pass

    data: List[Dict[str, str]] = json.loads(Path(args.input_json).read_text())
    if args.n_samples is not None:
        data = data[: args.n_samples]

    flat_queue, results_map = flatten_dataset(data)

    completed_pairs = set()
    if Path(args.output_json).exists():
        try:
            existing_items = json.loads(Path(args.output_json).read_text())
            for item in existing_items:
                prompt_count = str(item["prompt_count"])
                if prompt_count in results_map:
                    results_map[prompt_count].update(item)
                for k, val in item.items():
                    if k == "prompt_count":
                        continue
                    if isinstance(val, str) and val.strip():
                        completed_pairs.add((prompt_count, k))
        except Exception as e:
            logging.warning("Could not load existing output (%s) - starting fresh", e)

    flat_queue = [t for t in flat_queue if (t[0], t[1]) not in completed_pairs]
    flat_queue.sort(key=lambda t: len(tokenizer(t[2]).input_ids))

    def _save_partial() -> None:
        Path(args.output_json).write_text(
            json.dumps(list(results_map.values()), indent=2, ensure_ascii=False)
        )

    def _handle_signal(sig_num, _frame):
        logging.info("Received signal %s - saving partial results and exiting", sig_num)
        _save_partial()
        flush_bucket(len(flat_queue), force=True)
#        wandb.finish()
        sys.exit(0)

    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _handle_signal)
    atexit.register(_save_partial)

    if args.pooling == "none":
        bytes_per_example = len(capture_layers) * args.max_tokens * hidden_size * 2
    else:
        bytes_per_example = len(capture_layers) * hidden_size * 2
    estimated_total = bytes_per_example * (args.n_samples or 550_000)
    logging.info("=== estimated raw activation size ~ %.1f GB ===", estimated_total / 1024**3)

    failures = []

    # inside the main loop
    for start in tqdm(range(0, len(flat_queue), args.batch), desc="generating"):
        batch_slice = flat_queue[start : start + args.batch]
        batch_ids, batch_keys, batch_texts = zip(*batch_slice)

        # Tokenise on CPU
        inputs = tokenizer(list(batch_texts), return_tensors="pt", padding=True)

        # Get prompt lengths *before* moving to GPU
        input_lens = inputs["attention_mask"].sum(dim=1)

        # Move every tensor to the model’s first device (works for single-GPU or device_map="auto")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with _INFER_CTX():
            gen_kwargs = dict(max_new_tokens=args.max_tokens, pad_token_id=tokenizer.eos_token_id)
            if args.temperature > 0:
                gen_kwargs.update(temperature=args.temperature, do_sample=True)
            else:
                gen_kwargs["do_sample"] = False
            outputs = model.generate(**inputs, **gen_kwargs)

        for i in range(len(batch_slice)):
            try:
                start_tok = int(input_lens[i])
                completion_ids = outputs[i, start_tok:]
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                results_map[batch_ids[i]][batch_keys[i]] = completion
            except Exception as e:
                failures.append((batch_ids[i], batch_keys[i], repr(e)))
                logging.exception("Generation failed for %s-%s", batch_ids[i], batch_keys[i])
        del inputs, outputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        global current_bytes
        batch_rows = len(batch_slice)
        for tens_list in bucket.values():
            for t in tens_list[-batch_rows:]:
                current_bytes += t.numel() * 2

        examples_in_current_shard += len(batch_slice)
        flush_bucket(start + len(batch_slice), force=False)
        if examples_in_current_shard >= args.shard_size:
            flush_bucket(start + len(batch_slice), force=True)
            examples_in_current_shard = 0
        if (start + len(batch_slice)) % args.log_every == 0:
            logging.info("Processed %d / %d prompts", start + len(batch_slice), len(flat_queue))

    Path(args.output_json).write_text(json.dumps(list(results_map.values()), indent=2, ensure_ascii=False))
    flush_bucket(len(flat_queue), force=True)
    #run_artifact = wandb.run.log_artifact(artifact)
    #run_artifact.wait()
    #for p in saved_shards:
    #    try:
    #        p.unlink()
    #    except FileNotFoundError:
    #        pass

#    if upload_acts:
#        wandb.log({"total_examples": len(flat_queue)})
#        wandb.finish()
    # else: no W&B upload, no cleanup needed
#    else:
#        pass

    if failures:
        print("\nUnhandled exceptions on %d prompts:" % len(failures))
        for pc, key, err in failures[:20]:
            print(f"  {pc}-{key}: {err[:120]}…")
    else:
        print("\nAll prompts completed without runtime errors 🎉")
    logging.info("Finished OK - wrote %d items to %s", len(results_map), args.output_json)

if __name__ == "__main__":
    main()
