#!/usr/bin/env python3
"""
<<<<<<< HEAD
=======
perplexity_after_ft.py
------------------------

Compute perplexity for every *paraphrase* in an Alpaca‑style JSON, but
**once for each LoRA/PEFT adapter you pass in**.  The script appends a new
field to each paraphrase whose key is provided by the user (one per adapter).

Example
-------
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
python perplexity_after_ft.py.py \
       f_finetune/data/all_alpaca_gemma-2-2b-it.json \
       f_analysis/alpaca_with_ppl.json \
       --model google/gemma-2b-it \
       --adapter \
           f_finetune/outputs/alpaca/all_layers/outputs_buckets_1-1/final=ft_buckets_1-1_alpaca_ppl \
       --adapter \
           f_finetune/outputs/alpaca/all_layers/outputs_buckets_1-2/final=ft_buckets_1-2_alpaca_ppl \
       --batch 16
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from huggingface_hub import HfApi, login as hf_login
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

<<<<<<< HEAD
# Optional deps
=======
# Optional deps ------------------------------------------------------------------
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
try:
    from transformers import BitsAndBytesConfig  # type: ignore
    _BITSANDBYTES_OK = True
except (ImportError, AttributeError):
    BitsAndBytesConfig = None  # type: ignore
    _BITSANDBYTES_OK = False

try:
    import importlib
    import flash_attn            # noqa: F401
    importlib.import_module("flash_attn.flash_attn_interface")
    _FLASH2_OK = True
except Exception:
    _FLASH2_OK = False

try:
    from peft import PeftModel, PeftConfig  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "peft is required for this script.  Install with:\n"
        "   python -m pip install peft"
    ) from exc

_INFER_CTX = getattr(torch, "inference_mode", torch.no_grad)


def ensure_hf_auth(token: str | None) -> None:
    if token:
        hf_login(token=token, add_to_git_credential=False, new_session=True)


def assert_model_access(model_id: str, token: str | None) -> None:
    try:
        HfApi().model_info(model_id, token=token)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Token doesn’t have access to `{model_id}`."
        ) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute perplexity with PEFT adapters")
    p.add_argument("input_json", help="Input dataset (.json)")
    p.add_argument("output_json", help="Output file (.json)")
    p.add_argument("--model", default="google/gemma-2b-it",
                   help="Base model on HF hub (default: google/gemma-2b-it)")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
    p.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    p.add_argument("--log_every", type=int, default=250)
    p.add_argument(
        "--adapter",
        action="append",
        metavar="PATH[=FIELD]",
        default=[],
        help=(
            "Path to a LoRA/PEFT adapter folder.  "
            "You may repeat the flag.  "
            "Optionally append '=field_name' to choose the JSON key "
            "that will store the perplexity (default: "
            "'ft_<basename>_ppl')."
        ),
    )
    p.add_argument(
        "--base_field_prefix",
        default="ft",
        help="Prefix for field names when none is supplied (default: 'ft')",
    )
    p.add_argument(
        "--debug_extra", "--debug-extra", type=int, default=0,
        help="If >0, print the first N texts in every batch together with "
             "token‑count, loss and PPL (useful for spotting anomalies).",
    )
    p.add_argument(
        "--debug_tokens", action="store_true",
        help="Dump token‑wise losses for the first --debug_extra sequences."
    )
    return p.parse_args()


def _parse_adapter_spec(raw: str, prefix: str, idx: int) -> Tuple[str, str]:
    """
<<<<<<< HEAD
    Parse 'path=field' or just 'path'  ->  (path, field)
=======
    Parse 'path=field' or just 'path'  →  (path, field)
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
    """
    if "=" in raw:
        path, field = raw.split("=", 1)
    else:
        path = raw
        field = f"{prefix}_{Path(path).stem}_ppl" if prefix else f"adapter_{idx}_ppl"
    return path, field


def load_model_and_tokenizer(
    model_id: str, device: str, quant: str
) -> Tuple[Any, Any, str]:
    model_kwargs: Dict[str, Any] = dict(device_map=device)

    if _FLASH2_OK:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if quant == "none":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif not _BITSANDBYTES_OK:
        logging.warning("bitsandbytes not available; switching to bf16")
        model_kwargs["torch_dtype"] = torch.bfloat16
        quant = "none"
    else:
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=(quant == "8bit"),
            load_in_4bit=(quant == "4bit"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = bnb_cfg

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception as e:
        if model_kwargs.pop("attn_implementation", None) == "flash_attention_2":
            logging.warning("flash‑attn failed (%s) – retrying without", e)
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        elif quant != "none":
            logging.warning("Quant load failed (%s) – retry bf16", e)
            model_kwargs = dict(device_map=device, torch_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            quant = "none"
        else:
            raise

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer, quant


<<<<<<< HEAD
=======
# ---------------------------------------------------------------------------  
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
def compute_batch_perplexity(
    model, tokenizer, texts: Sequence[str]
) -> List[float]:
    """
    Robust per‑sample perplexity that works no matter what shape `out.loss` has.

    * Ignores padding tokens.
    * Never returns scalar when batch>1.
    * Works for any Transformers version (pre/post 4.40).
    """
    enc = tokenizer(texts, return_tensors="pt", padding=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    # Mask pad tokens in labels so they don't contribute to loss
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100

    with torch.inference_mode():
        out = model(**enc, labels=labels)

<<<<<<< HEAD
    # Case A: new Transformers -> out.loss shape = (batch,)
    if out.loss.dim() == 1 and out.loss.size(0) == len(texts):
        seq_loss = out.loss
    else:
        # Case B: scalar loss -> compute token‑wise and average manually
=======
    # Case A: new Transformers → out.loss shape = (batch,)
    if out.loss.dim() == 1 and out.loss.size(0) == len(texts):
        seq_loss = out.loss
    else:
        # Case B: scalar loss → compute token‑wise and average manually
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
        logits = out.logits[:, :-1].contiguous()
        tgt    = labels[:, 1:].contiguous()

        ce = torch.nn.CrossEntropyLoss(reduction="none")
        tok_loss = ce(
            logits.view(-1, logits.size(-1)),
            tgt.view(-1),
        ).view(tgt.size())

        valid_tok_mask = (tgt != -100)
        seq_loss = (tok_loss * valid_tok_mask).sum(1) / valid_tok_mask.sum(1)

    seq_loss = seq_loss.clamp(max=50)           # keep exp stable
    ppl = torch.exp(seq_loss)


<<<<<<< HEAD
    # Optional verbose dump
=======
    # Optional verbose dump -------------------------------------------------
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
    dbg_left  = getattr(compute_batch_perplexity, "_dbg_left", 0)
    dbg_tok   = getattr(compute_batch_perplexity, "_dbg_tokens", False)
    if dbg_left > 0:
        for t, l, pv in zip(texts[:dbg_left], seq_loss, ppl):
            ntok = len(tokenizer(t).input_ids)
            logging.warning(
                "[DEBUG] tok=%3d  loss=%6.3f  ppl=%10.2f  %.80s",
                ntok, l.item(), pv.item(), t.replace("\n", " ")
            )

            if dbg_tok:
                # Per‑token inspection
                ids = tokenizer(t, return_tensors="pt").input_ids.to(model.device)
                with torch.inference_mode():
                    lp = torch.log_softmax(model(ids).logits[0, :-1], dim=-1)
                for idx, tok_id in enumerate(ids[0, 1:]):   # skip BOS
                    token = tokenizer.convert_ids_to_tokens(int(tok_id))
                    ce    = -lp[idx, tok_id].item()
                    logging.warning(
                        "        %02d  %-12s  CE=%6.3f", idx, token, ce
                    )
        compute_batch_perplexity._dbg_left = max(0, dbg_left - len(texts))


    return ppl.cpu().tolist()
<<<<<<< HEAD
=======
# ---------------------------------------------------------------------------
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab


def main() -> None:
    args = parse_args()

    if args.debug_extra > 0:
        # make the quota available inside compute_batch_perplexity
        compute_batch_perplexity._dbg_left = args.debug_extra
        compute_batch_perplexity._dbg_tokens = args.debug_tokens
        
    # Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/run_ppl_ft_{Path(args.input_json).stem}_{timestamp}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    logging.info("==== run started ====")
    logging.info("cmd: %s", " ".join(sys.argv))

    # Auth / model access
    ensure_hf_auth(args.hf_token)
    assert_model_access(args.model, args.hf_token)

    # Prepare adapters
    if not args.adapter:
        logging.error("You must pass at least one --adapter.")
        sys.exit("Error: --adapter is required")

    adapters: List[Tuple[str, str]] = [
        _parse_adapter_spec(raw, args.base_field_prefix, i + 1)
        for i, raw in enumerate(args.adapter)
    ]
    logging.info("Adapters to evaluate: %s", adapters)

    # Base model + tokenizer
    print("Loading base model – may download a few GB …")
    base_model, tokenizer, _ = load_model_and_tokenizer(
        args.model, args.device, args.quant
    )

    # Load data
    data: List[Dict[str, Any]] = json.loads(Path(args.input_json).read_text())
    if args.n_samples:
        data = data[: args.n_samples]

    total_phrases = sum(len(r["paraphrases"]) for r in data)
    logging.info("Dataset: %d records – %d paraphrases", len(data), total_phrases)

    # Shutdown hooks
    def _save():
        Path(args.output_json).write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )

    def _sig_handler(sig, _frame):
        logging.warning("Signal %d caught – saving & exiting", sig)
        _save()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _sig_handler)
    atexit.register(_save)

    # Actual evaluation
    for adapter_idx, (adapter_path, field_name) in enumerate(adapters, 1):
        logging.info("=== Adapter %d / %d – %s ===", adapter_idx, len(adapters), adapter_path)
<<<<<<< HEAD
        print(f"Adapter {adapter_idx}/{len(adapters)} -> {field_name}")
=======
        print(f"Adapter {adapter_idx}/{len(adapters)} → {field_name}")
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab

        # Attach LoRA
        try:
            model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
        except Exception as e:  # noqa: BLE001
            logging.exception("Could not load adapter %s (%s)", adapter_path, e)
            continue
        model.eval()

        processed, err, ppl_sum = 0, 0, 0.0
        buffer: List[Tuple[int, int, str]] = []

        for rec_idx, record in enumerate(tqdm(data, desc="collecting")):
            for para_idx, para in enumerate(record["paraphrases"]):
                buffer.append((rec_idx, para_idx, para["paraphrase"]))
                if len(buffer) >= args.batch:
                    recs, pris, texts = zip(*buffer)
                    try:
                        ppls = compute_batch_perplexity(model, tokenizer, list(texts))
                    except Exception as e:  # noqa: BLE001
                        logging.exception("Perplexity batch failed: %s", e)
                        err += len(texts)
                        buffer.clear()
                        continue

                    for r, p, v in zip(recs, pris, ppls):
                        data[r]["paraphrases"][p][field_name] = v
                        processed += 1
                        ppl_sum += v
                    buffer.clear()

                    if processed and processed % args.log_every == 0:
                        logging.info("Adapter %s: %d / %d done", field_name, processed, total_phrases)
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        gc.collect()

        # Flush remainder
        if buffer:
            recs, pris, texts = zip(*buffer)
            try:
                ppls = compute_batch_perplexity(model, tokenizer, list(texts))
            except Exception as e:  # noqa: BLE001
                logging.exception("Perplexity batch failed: %s", e)
                err += len(texts)
            else:
                for r, p, v in zip(recs, pris, ppls):
                    data[r]["paraphrases"][p][field_name] = v
                    processed += 1
                    ppl_sum += v

        avg = ppl_sum / max(processed, 1)
        logging.info("Adapter %s – finished: ok=%d  err=%d  avg_ppl=%.3f",
                     field_name, processed, err, avg)
        print(f"Adapter {field_name}: {processed} OK / {err} errors – avg ppl {avg:.2f}")

        # Free LoRA weights (but keep base model in memory)
        del model
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    _save()
<<<<<<< HEAD
    print(f"All adapters done -> saved to {args.output_json}")
=======
    print(f"All adapters done → saved to {args.output_json}")
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
    logging.info("All adapters done – output written to %s", args.output_json)


if __name__ == "__main__":
    main()
