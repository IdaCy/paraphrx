#!/usr/bin/env python3
"""
<<<<<<< HEAD
=======
compute_perplexity.py

Add a `perplexity` score (Gemma‑2‑2B‑IT by default) to every paraphrase entry
in an Alpaca‑style dataset.

Example
-------
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
python compute_perplexity.py \
       f_finetune/data/all_alpaca_gemma-2-2b-it.json \
       f_analysis/data/alpaca_with_ppl.json \
       --batch 16 --quant 4bit --log_every 500
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
from typing import Any, Dict, List, Tuple

import torch
from huggingface_hub import HfApi, login as hf_login
from tqdm import tqdm
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
)

<<<<<<< HEAD
# Optional deps
=======
# Optional deps -------------------------------------------------------------
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
except Exception:                # pragma: no cover
    _FLASH2_OK = False
<<<<<<< HEAD
=======
# --------------------------------------------------------------------------
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab

_INFER_CTX = getattr(torch, "inference_mode", torch.no_grad)


def ensure_hf_auth(token: str | None) -> None:
    if token:
        hf_login(token=token, add_to_git_credential=False, new_session=True)


def assert_model_access(model_id: str, token: str | None) -> None:
    try:
        HfApi().model_info(model_id, token=token)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Token does not have access to `{model_id}` – did you accept the license?"
        ) from e


<<<<<<< HEAD
=======
# ---------------------------------------------------------------------------


>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute perplexity for Alpaca paraphrases")
    p.add_argument("input_json", help="Path to input dataset (.json)")
    p.add_argument("output_json", help="File to write the augmented dataset (.json)")
    p.add_argument(
        "--model",
        default="google/gemma-2b-it",
        help="HF model repo (default: google/gemma-2b-it)",
    )
    p.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n_samples", type=int, default=None)
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
    p.add_argument("--log_every", type=int, default=200)
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


<<<<<<< HEAD
=======
# ---------------------------------------------------------------------------


>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
def load_model_and_tokenizer(
    model_id: str, device: str, quant: str
) -> Tuple[Any, Any, str]:
    model_kwargs: Dict[str, Any] = dict(device_map=device)

    if _FLASH2_OK:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if quant == "none":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif not _BITSANDBYTES_OK:
        logging.warning("bitsandbytes unavailable – reverting to bf16")
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

    # Try load – fall back if flash/quant fails
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    except Exception as e:
        if model_kwargs.pop("attn_implementation", None) == "flash_attention_2":
            logging.warning("flash_attn failed (%s) – retrying without", e)
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        elif quant != "none":
            logging.warning("Quant load failed (%s) – retrying bf16", e)
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

    # Case A: new Transformers → out.loss shape = (batch,)
    if out.loss.dim() == 1 and out.loss.size(0) == len(texts):
        seq_loss = out.loss
    else:
        # Case B: scalar loss → compute token‑wise and average manually
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

<<<<<<< HEAD
    # Logging
=======
    # Logging ---------------------------------------------------------------
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        f"logs/run_ppl_{Path(args.input_json).stem}_"
        f"{args.model.replace('/','-')}_{timestamp}.log"
    )
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    logging.info("==== perplexity run started ====")
    logging.info("cmd args: %s", vars(args))

<<<<<<< HEAD
    # Auth & HF access
    ensure_hf_auth(args.hf_token)
    assert_model_access(args.model, args.hf_token)

    # Model
=======
    # Auth & HF access ------------------------------------------------------
    ensure_hf_auth(args.hf_token)
    assert_model_access(args.model, args.hf_token)

    # Model -----------------------------------------------------------------
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
    print("Loading tokenizer & model – first run may download a few GB …")
    model, tokenizer, actual_quant = load_model_and_tokenizer(
        args.model, args.device, args.quant
    )
    if actual_quant != args.quant:
        logging.info("Quantisation changed to %s due to fall‑back", actual_quant)

<<<<<<< HEAD
    # Data
=======
    # Data ------------------------------------------------------------------
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
    data: List[Dict[str, Any]] = json.loads(Path(args.input_json).read_text())
    if args.n_samples:
        data = data[: args.n_samples]

    total_paraphrases = sum(len(x["paraphrases"]) for x in data)
    logging.info("Loaded %d records – %d paraphrases", len(data), total_paraphrases)

    # Summary stats for the tail report
    stats = dict(processed=0, errors=0, ppl_sum=0.0)

<<<<<<< HEAD
    # Graceful shutdown
=======
    # Graceful shutdown -----------------------------------------------------
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
    def _save_partial() -> None:
        Path(args.output_json).write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )

    def _signal_handler(sig_num, _frame):
        logging.warning("Received signal %d – saving partial results & exiting", sig_num)
        _save_partial()
        sys.exit(0)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _signal_handler)
    atexit.register(_save_partial)

<<<<<<< HEAD
=======
    # ----------------------------------------------------------------------------
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
    batch_buffer: List[Tuple[int, int, str]] = []  # record‑idx, paraphrase‑idx, text

    for rec_idx, record in enumerate(tqdm(data, desc="collecting")):
        for p_idx, para in enumerate(record["paraphrases"]):
            text = para["paraphrase"]
            batch_buffer.append((rec_idx, p_idx, text))

            if len(batch_buffer) >= args.batch:
                # Compute PPL for the batch
                rec_ids, para_ids, texts = zip(*batch_buffer)
                try:
                    ppls = compute_batch_perplexity(model, tokenizer, list(texts))
                except Exception as e:  # noqa: BLE001
                    logging.exception("Error computing perplexity: %s", e)
                    stats["errors"] += len(texts)
                    batch_buffer.clear()
                    continue

                for r_id, p_id, ppl in zip(rec_ids, para_ids, ppls):
                    data[r_id]["paraphrases"][p_id]["perplexity"] = ppl
                    stats["processed"] += 1
                    stats["ppl_sum"] += ppl

                batch_buffer.clear()

                # Maintenance
                if stats["processed"] and stats["processed"] % args.log_every == 0:
                    logging.info("Processed %d / %d paraphrases",
                                 stats["processed"], total_paraphrases)
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    gc.collect()

<<<<<<< HEAD
    # Flush remaining buffer
=======
    # Flush remaining buffer -----------------------------------------------------
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
    if batch_buffer:
        rec_ids, para_ids, texts = zip(*batch_buffer)
        try:
            ppls = compute_batch_perplexity(model, tokenizer, list(texts))
        except Exception as e:  # noqa: BLE001
            logging.exception("Error computing perplexity: %s", e)
            stats["errors"] += len(texts)
        else:
            for r_id, p_id, ppl in zip(rec_ids, para_ids, ppls):
                data[r_id]["paraphrases"][p_id]["perplexity"] = ppl
                stats["processed"] += 1
                stats["ppl_sum"] += ppl

    avg_ppl = stats["ppl_sum"] / max(stats["processed"], 1)
    tail_report = (
        f"\n=== Run summary ({datetime.now():%Y-%m-%d %H:%M:%S}) ===\n"
        f"Records processed  : {len(data)}\n"
        f"Paraphrases OK     : {stats['processed']}\n"
        f"Errors             : {stats['errors']}\n"
        f"Avg perplexity     : {avg_ppl:.3f}\n"
    )
    logging.info(tail_report.strip().replace("\n", " | "))
    print(tail_report)

<<<<<<< HEAD
    # Persist to disk
=======
    # Persist to disk --------------------------------------------------------------
>>>>>>> 583483815076ac50f6910be9fc71b4d9ef76c5ab
    Path(args.output_json).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved → {args.output_json}")
    logging.info("Finished OK – wrote %s", args.output_json)


if __name__ == "__main__":
    main()
