#!/usr/bin/env python3
"""
SCRIPT C

causal_patch_and_answers.py

Purpose:
- Build BASE and FT models (FT can be a merged model dir or a local LoRA adapter, which is merged into BASE).
- Construct *hybrid* models at a chosen layer (default 6):
    * HYB_ATTN: BASE with FT **attention** at that layer.
    * HYB_MLP:  BASE with FT **MLP** at that layer.
    * HYB_BOTH: BASE with FT **attention+MLP** at that layer.
- Run **inference** on the SECOND dataset (prompts JSON with instruction_original + instruct_* keys) for:
    BASE, FT, HYB_ATTN, HYB_MLP, HYB_BOTH.
- Emit **answers-JSON** for each variant in the exact layout you use for AlpacaEval.
- Produce a compact numeric summary (CSV) and sanity plots (answer length distributions).
- NEW: Batching for speed; NEW comparative plots that overlay variants in the same figure.

Kept from your tuned setup:
- Comprehensive logging; seed=42 default.
- Flexible model loading & LoRA merging; device and dtype handling.
- Layer index flag; paraphrase key filtering; max prompts limit; safe fallbacks.
- Deterministic generation defaults (configurable).

Outputs (to --outdir):
- answers_BASE.json
- answers_FT_{TAG}.json
- answers_HYB_ATTN_{TAG}.json
- answers_HYB_MLP_{TAG}.json
- answers_HYB_BOTH_{TAG}.json
- generations_summary_{TAG}.csv  (prompt_count, key, n_chars, n_tokens_est)
- plots (existing):
  - answer_length_hist_{VARIANT}.png
- NEW comparative plots:
  - answer_length_hist_overlay_{TAG}.png               (all variants in one histogram)
  - answer_length_box_overlay_{TAG}.png                (boxplot of lengths across variants)
  - answer_length_scatter_base_vs_variants_{TAG}.png   (x=BASE length, y=variant length, all variants colored)
  - length_closeness_to_FT_bars_{TAG}.png              (|len(variant)-len(FT)| mean±CI; component attribution-ish)
  - instruction_sensitivity_bars_{TAG}.png             (per-prompt length std across paraphrases; mean±CI by variant)
  - instruction_sensitivity_box_{TAG}.png              (boxplot of per-prompt length std across variants)
- README.md (what to run in AlpacaEval)
"""

from __future__ import annotations
import argparse, json, logging, os, re, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# Utilities

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

# Data (SECOND prompts)

@dataclass
class Item:
    prompt_count: int
    instruction_original: str
    paraphrases: Dict[str, str]
    input_text: str

def load_instruction_json(path: str | Path, keys: Optional[List[str]] = None, keys_regex: Optional[str] = None) -> List[Item]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items: List[Item] = []
    pat = re.compile(keys_regex) if keys_regex else None
    for obj in data:
        pc = int(obj["prompt_count"])
        inp = obj.get("input","") or ""
        ph = {}
        for k,v in obj.items():
            if not isinstance(v, str): continue
            if not k.startswith("instruct_"): continue
            if keys and k not in keys: continue
            if pat and pat.search(k) is None: continue
            ph[k] = v
        items.append(Item(
            prompt_count=pc,
            instruction_original=obj.get("instruction_original", ""),
            paraphrases=ph,
            input_text=inp
        ))
    return items

def build_prompt_text(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return f"{instruction}\n\nInput: {input_text}"
    return instruction

# Model access & loader

class BlockAccessor:
    def __init__(self, model: nn.Module, layer_index: int):
        self.model = model
        self.layer_index = layer_index
        # Try common architectures
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.layers = model.transformer.h
        else:
            raise TypeError("Unsupported model architecture. Expected .model.layers or .transformer.h")
        self.block = self.layers[layer_index]
        # attention
        self.attn = getattr(self.block, "self_attn", None) or getattr(self.block, "attention", None)
        # mlp
        self.mlp = getattr(self.block, "mlp", None) or getattr(self.block, "feed_forward", None)
        if self.attn is None or self.mlp is None:
            raise TypeError("Could not access attention or MLP submodule at the given layer.")

def _is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() or \
           len(list(path.glob("adapter_model*.bin"))) > 0 or \
           len(list(path.glob("adapter_model*.safetensors"))) > 0

def _is_merged_model_dir(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    has_weights = any((path / n).exists() for n in ["pytorch_model.bin", "pytorch_model.pt", "model.bin"]) \
                  or len(list(path.glob("pytorch_model-*.bin"))) > 0 \
                  or (path / "model.safetensors").exists() \
                  or len(list(path.glob("model-*.safetensors"))) > 0
    return has_weights

def build_model_and_tokenizer(
    base_model_name_or_path: str,
    ft_model_name_or_path: Optional[str],
    ft_lora_adapter: Optional[str],
    device: str,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    base = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device).eval()

    if ft_model_name_or_path is None and ft_lora_adapter is None:
        raise ValueError("Provide --ft_model_name_or_path (merged FT) OR --ft_lora_adapter (LoRA adapter dir).")

    if ft_lora_adapter is not None:
        apath = Path(ft_lora_adapter)
        if apath.exists() and _is_adapter_dir(apath):
            if not _HAS_PEFT:
                raise RuntimeError("peft not installed but --ft_lora_adapter was provided.")
            ft = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
            ft = PeftModel.from_pretrained(ft, str(apath))
            ft = ft.merge_and_unload().eval()
            return base, ft, tokenizer
        # if the adapter path is actually a merged dir:
        if apath.exists() and _is_merged_model_dir(apath):
            ft = AutoModelForCausalLM.from_pretrained(str(apath), torch_dtype=dtype).to(device).eval()
            return base, ft, tokenizer

    if ft_model_name_or_path is None:
        raise ValueError("When --ft_lora_adapter is not a local adapter dir, you must provide --ft_model_name_or_path.")
    ft = AutoModelForCausalLM.from_pretrained(ft_model_name_or_path, torch_dtype=dtype).to(device).eval()
    return base, ft, tokenizer

# Weight swapping helpers

def copy_module_weights(dst: nn.Module, src: nn.Module):
    with torch.no_grad():
        for (n_d, p_d), (n_s, p_s) in zip(dst.named_parameters(), src.named_parameters()):
            if p_d.shape == p_s.shape:
                p_d.copy_(p_s)
        for (n_d, b_d), (n_s, b_s) in zip(dst.named_buffers(), src.named_buffers()):
            if b_d.shape == b_s.shape:
                b_d.copy_(b_s)

class SwapContext:
    """
    Context manager for temporarily swapping submodules from FT into BASE
    """
    def __init__(self, base: nn.Module, ft: nn.Module, layer_index: int, swap_attn: bool=False, swap_mlp: bool=False):
        self.base = base; self.ft = ft; self.layer_index = layer_index
        self.swap_attn = swap_attn; self.swap_mlp = swap_mlp
        self._backup = {}

    def __enter__(self):
        ba = BlockAccessor(self.base, self.layer_index)
        fa = BlockAccessor(self.ft,   self.layer_index)
        if self.swap_attn:
            self._backup["attn"] = BlockAccessor(self.base, self.layer_index).attn.state_dict()
            copy_module_weights(ba.attn, fa.attn)
        if self.swap_mlp:
            self._backup["mlp"] = BlockAccessor(self.base, self.layer_index).mlp.state_dict()
            copy_module_weights(ba.mlp, fa.mlp)
        return self.base

    def __exit__(self, exc_type, exc, tb):
        ba = BlockAccessor(self.base, self.layer_index)
        if "attn" in self._backup:
            ba.attn.load_state_dict(self._backup["attn"])
        if "mlp" in self._backup:
            ba.mlp.load_state_dict(self._backup["mlp"])
        self._backup.clear()
        return False

# --------
# Generation (batched)
# --------

def _generate_batch(
    model, tokenizer, prompts: List[str], device: str,
    max_new_tokens: int, temperature: float, top_p: float, do_sample: bool
) -> List[str]:
    """
    Generate continuations for a list of prompts using a single batched call.
    Returns list of strings containing only the continuation (prompt stripped).
    """
    if len(prompts) == 0:
        return []
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(device)
    input_lengths = [int(l) for l in (enc["attention_mask"].sum(dim=1).tolist())]
    gen_out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    results = []
    for i in range(gen_out.shape[0]):
        full_ids = gen_out[i].tolist()
        cont_ids = full_ids[input_lengths[i]:]
        text = tokenizer.decode(cont_ids, skip_special_tokens=True).strip()
        results.append(text)
    return results

@torch.inference_mode()
def generate_answers_batched(
    model, tokenizer, items: List[Item], device: str,
    max_new_tokens: int, temperature: float, top_p: float, do_sample: bool,
    batch_size: int, log: logging.Logger
) -> List[dict]:
    """
    Vectorized generator:
      - flattens all (prompt_count, key, text) tasks
      - generates in batches
      - reassembles per-item records in the required schema
    """
    # Build flat task list
    flat: List[Tuple[int, str, str]] = []  # (prompt_count, key, prompt_text)
    # We store the order per item so reconstruction is easy
    per_item_layout: Dict[int, List[str]] = {}  # prompt_count -> list of keys in order we will fill
    for it in items:
        keys_order = ["instruction_original"] + sorted(it.paraphrases.keys())
        per_item_layout[it.prompt_count] = keys_order
        # instruction_original
        flat.append((it.prompt_count, "instruction_original", build_prompt_text(it.instruction_original, it.input_text)))
        # paraphrases (sorted, to be deterministic)
        for k in sorted(it.paraphrases.keys()):
            flat.append((it.prompt_count, k, build_prompt_text(it.paraphrases[k], it.input_text)))

    # Batched generation
    outputs: List[str] = []
    N = len(flat)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        prompts = [flat[i][2] for i in range(start, end)]
        batch_out = _generate_batch(model, tokenizer, prompts, device, max_new_tokens, temperature, top_p, do_sample)
        outputs.extend(batch_out)
        if ((start // batch_size) % 10) == 0:
            log.info("  batch %d/%d", start // batch_size + 1, (N + batch_size - 1) // batch_size)

    # Reassemble per-item records
    idx = 0
    records: Dict[int, Dict[str, str]] = {}
    for it in items:
        layout = per_item_layout[it.prompt_count]
        rec = {"prompt_count": it.prompt_count}
        for key in layout:
            rec[key] = outputs[idx]; idx += 1
        records[it.prompt_count] = rec

    # Preserve original item order
    return [records[it.prompt_count] for it in items]

def build_answers_json_record(item: Item, mapping: Dict[str, str]) -> dict:
    rec = {"prompt_count": item.prompt_count}
    for k, v in mapping.items():
        rec[k] = v
    return rec

# Main

def main():
    ap = argparse.ArgumentParser(description="Causal weight-patching + answers JSON emission (batched + comparative plots)")
    ap.add_argument("--instructions_json", type=str, required=True, help="SECOND dataset prompts JSON")
    ap.add_argument("--base_model_name_or_path", type=str, required=True)
    ap.add_argument("--ft_model_name_or_path", type=str, default=None)
    ap.add_argument("--ft_lora_adapter", type=str, default=None)
    ap.add_argument("--layer_index", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="FT", help="Label for the FT run, e.g., SFT or LAP")

    # filters / scaling
    ap.add_argument("--max_prompts", type=int, default=None, help="Optional cap on number of prompts processed")
    ap.add_argument("--keys", type=str, default=None, help="Comma-separated instruct_* keys to include")
    ap.add_argument("--keys_regex", type=str, default=None, help="Regex to filter instruct_* keys")

    # decoding
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--do_sample", action="store_true")

    # batching
    ap.add_argument("--batch_size", type=int, default=8)

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("causal_patch")

    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    # Load data
    keys = [k.strip() for k in args.keys.split(",")] if args.keys else None
    items = load_instruction_json(args.instructions_json, keys=keys, keys_regex=args.keys_regex)
    if args.max_prompts is not None:
        items = items[:args.max_prompts]
    log.info("Loaded %d prompts from SECOND dataset.", len(items))

    # Load models
    base, ft, tokenizer = build_model_and_tokenizer(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, args.device
    )
    log.info("Models loaded. Layer index for swapping: %d", args.layer_index)

    # Prepare holders for outputs
    answers_BASE = []
    answers_FT   = []
    answers_HYB_ATTN = []
    answers_HYB_MLP  = []
    answers_HYB_BOTH = []

    # Summary rows for quick sanity checks
    summ_rows = []

    # ----------- Generate (batched) for each variant -----------
    def collect_summary(variant: str, records: List[dict]):
        for rec in records:
            pc = rec["prompt_count"]
            for k,v in rec.items():
                if k == "prompt_count": continue
                summ_rows.append({
                    "variant": variant,
                    "prompt_count": pc,
                    "key": k,
                    "n_chars": len(v),
                    "n_tokens_est": len(tokenizer.encode(v))
                })

    # BASE
    log.info("Generating with BASE (batched, bs=%d)...", args.batch_size)
    t0 = time.time()
    answers_BASE = generate_answers_batched(
        base, tokenizer, items, args.device,
        args.max_new_tokens, args.temperature, args.top_p, args.do_sample,
        args.batch_size, log
    )
    collect_summary("BASE", answers_BASE)
    log.info("BASE done in %.1fs", time.time()-t0)

    # FT
    log.info("Generating with FT (batched, bs=%d)...", args.batch_size)
    t0 = time.time()
    answers_FT = generate_answers_batched(
        ft, tokenizer, items, args.device,
        args.max_new_tokens, args.temperature, args.top_p, args.do_sample,
        args.batch_size, log
    )
    collect_summary(f"FT_{args.tag}", answers_FT)
    log.info("FT done in %.1fs", time.time()-t0)

    # HYB_ATTN
    log.info("Generating with HYB_ATTN (BASE + FT attention @ layer %d)...", args.layer_index)
    t0=time.time()
    with SwapContext(base, ft, args.layer_index, swap_attn=True, swap_mlp=False) as hyb:
        answers_HYB_ATTN = generate_answers_batched(
            hyb, tokenizer, items, args.device,
            args.max_new_tokens, args.temperature, args.top_p, args.do_sample,
            args.batch_size, log
        )
    collect_summary(f"HYB_ATTN_{args.tag}", answers_HYB_ATTN)
    log.info("HYB_ATTN done in %.1fs", time.time()-t0)

    # HYB_MLP
    log.info("Generating with HYB_MLP (BASE + FT MLP @ layer %d)...", args.layer_index)
    t0=time.time()
    with SwapContext(base, ft, args.layer_index, swap_attn=False, swap_mlp=True) as hyb:
        answers_HYB_MLP = generate_answers_batched(
            hyb, tokenizer, items, args.device,
            args.max_new_tokens, args.temperature, args.top_p, args.do_sample,
            args.batch_size, log
        )
    collect_summary(f"HYB_MLP_{args.tag}", answers_HYB_MLP)
    log.info("HYB_MLP done in %.1fs", time.time()-t0)

    # HYB_BOTH
    log.info("Generating with HYB_BOTH (BASE + FT attn+mlp @ layer %d)...", args.layer_index)
    t0=time.time()
    with SwapContext(base, ft, args.layer_index, swap_attn=True, swap_mlp=True) as hyb:
        answers_HYB_BOTH = generate_answers_batched(
            hyb, tokenizer, items, args.device,
            args.max_new_tokens, args.temperature, args.top_p, args.do_sample,
            args.batch_size, log
        )
    collect_summary(f"HYB_BOTH_{args.tag}", answers_HYB_BOTH)
    log.info("HYB_BOTH done in %.1fs", time.time()-t0)

    # Write answers JSONs
    (outdir / "answers_BASE.json").write_text(json.dumps(answers_BASE, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / f"answers_FT_{args.tag}.json").write_text(json.dumps(answers_FT, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / f"answers_HYB_ATTN_{args.tag}.json").write_text(json.dumps(answers_HYB_ATTN, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / f"answers_HYB_MLP_{args.tag}.json").write_text(json.dumps(answers_HYB_MLP, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / f"answers_HYB_BOTH_{args.tag}.json").write_text(json.dumps(answers_HYB_BOTH, ensure_ascii=False, indent=2), encoding="utf-8")

    # Summary CSV
    summ_df = pd.DataFrame(summ_rows)
    summ_df.to_csv(outdir / f"generations_summary_{args.tag}.csv", index=False)

    # Existing per-variant histograms
    for variant in sorted(summ_df["variant"].unique()):
        sdf = summ_df[summ_df["variant"] == variant]
        if sdf.empty: continue
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(sdf["n_tokens_est"], bins=40)
        ax.set_title(f"Answer length (tokens est.) — {variant}")
        ax.set_xlabel("tokens (approx)")
        ax.set_ylabel("# of (prompt,key) generations")
        fig.tight_layout()
        fig.savefig(outdir / f"answer_length_hist_{variant}.png", dpi=160)
        plt.close(fig)

    # -----------------------
    # NEW comparative graphics
    # -----------------------

    # 1) Overlayed histograms across variants
    try:
        fig, ax = plt.subplots(figsize=(10,5))
        for variant in ["BASE", f"FT_{args.tag}", f"HYB_ATTN_{args.tag}", f"HYB_MLP_{args.tag}", f"HYB_BOTH_{args.tag}"]:
            sdf = summ_df[summ_df["variant"] == variant]
            if sdf.empty: continue
            ax.hist(sdf["n_tokens_est"], bins=50, histtype="step", linewidth=1.5, label=variant, alpha=0.9, density=False)
        ax.set_title("Answer length distribution (overlay, tokens est.)")
        ax.set_xlabel("tokens (approx)")
        ax.set_ylabel("# generations")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"answer_length_hist_overlay_{args.tag}.png", dpi=170)
        plt.close(fig)
    except Exception as e:
        logging.warning("Overlay histogram failed: %s", e)

    # 2) Boxplot of lengths across variants in one figure
    try:
        fig, ax = plt.subplots(figsize=(10,5))
        data = []
        labels = []
        for variant in ["BASE", f"FT_{args.tag}", f"HYB_ATTN_{args.tag}", f"HYB_MLP_{args.tag}", f"HYB_BOTH_{args.tag}"]:
            vals = summ_df.loc[summ_df["variant"] == variant, "n_tokens_est"].values
            if len(vals) == 0: continue
            data.append(vals); labels.append(variant)
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
            ax.set_title("Answer length distributions (boxplot)")
            ax.set_ylabel("tokens (approx)")
            fig.tight_layout()
            fig.savefig(outdir / f"answer_length_box_overlay_{args.tag}.png", dpi=170)
        plt.close(fig)
    except Exception as e:
        logging.warning("Box overlay failed: %s", e)

    # 3) Scatter: BASE vs other variants (lengths), all colored in one plot
    try:
        base_map = summ_df[summ_df["variant"] == "BASE"][["prompt_count","key","n_tokens_est"]].rename(columns={"n_tokens_est":"base_len"})
        fig, ax = plt.subplots(figsize=(6,6))
        plotted=False
        for variant in [f"FT_{args.tag}", f"HYB_ATTN_{args.tag}", f"HYB_MLP_{args.tag}", f"HYB_BOTH_{args.tag}"]:
            vdf = summ_df[summ_df["variant"] == variant][["prompt_count","key","n_tokens_est"]].rename(columns={"n_tokens_est":"var_len"})
            m = base_map.merge(vdf, on=["prompt_count","key"])
            if m.empty: continue
            ax.scatter(m["base_len"].values, m["var_len"].values, s=8, alpha=0.5, label=variant)
            plotted=True
        if plotted:
            mn = 0
            mx = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([mn,mx],[mn,mx], linestyle="--", linewidth=1)
            ax.set_xlabel("BASE length (tokens est.)")
            ax.set_ylabel("Variant length (tokens est.)")
            ax.set_title("BASE vs variants (answer length)")
            ax.legend(markerscale=2)
            fig.tight_layout()
            fig.savefig(outdir / f"answer_length_scatter_base_vs_variants_{args.tag}.png", dpi=170)
        plt.close(fig)
    except Exception as e:
        logging.warning("Length scatter failed: %s", e)

    # 4) Component attribution-ish: which hybrid is closer to FT? (by length)
    #    Compute mean |len(variant) - len(FT)|.
    try:
        ft_map = summ_df[summ_df["variant"] == f"FT_{args.tag}"][["prompt_count","key","n_tokens_est"]].rename(columns={"n_tokens_est":"ft_len"})
        closeness = []
        for variant in ["BASE", f"HYB_ATTN_{args.tag}", f"HYB_MLP_{args.tag}", f"HYB_BOTH_{args.tag}"]:
            vdf = summ_df[summ_df["variant"] == variant][["prompt_count","key","n_tokens_est"]].rename(columns={"n_tokens_est":"var_len"})
            m = ft_map.merge(vdf, on=["prompt_count","key"])
            if m.empty: continue
            diffs = np.abs(m["var_len"].values - m["ft_len"].values)
            mean = float(diffs.mean()) if diffs.size>0 else 0.0
            std = float(diffs.std(ddof=1)) if diffs.size>1 else 0.0
            ci = 1.96 * std / np.sqrt(max(diffs.size,1)) if diffs.size>1 else 0.0
            closeness.append((variant, mean, ci))
        if closeness:
            fig, ax = plt.subplots(figsize=(8,4))
            labels = [c[0] for c in closeness]
            means  = [c[1] for c in closeness]
            cis    = [c[2] for c in closeness]
            ax.bar(labels, means, yerr=cis)
            ax.set_ylabel("Mean |len(variant) − len(FT)| (tokens)")
            ax.set_title("Closeness to FT by length (lower is better)")
            fig.tight_layout()
            fig.savefig(outdir / f"length_closeness_to_FT_bars_{args.tag}.png", dpi=170)
            plt.close(fig)
    except Exception as e:
        logging.warning("Closeness-to-FT plot failed: %s", e)

    # 5) Instruction sensitivity: per-prompt std over paraphrases (incl. instruction_original)
    #    For each variant and prompt_count, compute std across keys; then aggregate across prompts.
    try:
        sens_rows = []
        for variant in ["BASE", f"FT_{args.tag}", f"HYB_ATTN_{args.tag}", f"HYB_MLP_{args.tag}", f"HYB_BOTH_{args.tag}"]:
            vdf = summ_df[summ_df["variant"] == variant]
            for pc, grp in vdf.groupby("prompt_count"):
                vals = grp["n_tokens_est"].values
                if len(vals) < 2:
                    continue
                sens = float(np.std(vals, ddof=1))
                sens_rows.append({"variant": variant, "prompt_count": pc, "length_std_across_instructions": sens})
        sens_df = pd.DataFrame(sens_rows)
        if not sens_df.empty:
            sens_df.to_csv(outdir / f"instruction_sensitivity_{args.tag}.csv", index=False)

            # Bars (mean±CI across prompts)
            fig, ax = plt.subplots(figsize=(9,4))
            labels=[]; means=[]; cis=[]
            for variant in ["BASE", f"FT_{args.tag}", f"HYB_ATTN_{args.tag}", f"HYB_MLP_{args.tag}", f"HYB_BOTH_{args.tag}"]:
                vals = sens_df.loc[sens_df["variant"]==variant, "length_std_across_instructions"].values
                if vals.size == 0: continue
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1)) if vals.size>1 else 0.0
                ci = 1.96 * s / np.sqrt(vals.size) if vals.size>1 else 0.0
                labels.append(variant); means.append(m); cis.append(ci)
            if labels:
                ax.bar(labels, means, yerr=cis)
                ax.set_title("Instruction sensitivity (std of answer length across paraphrases)")
                ax.set_ylabel("Std (tokens)")
                fig.tight_layout()
                fig.savefig(outdir / f"instruction_sensitivity_bars_{args.tag}.png", dpi=170)
                plt.close(fig)

            # Boxplot
            fig, ax = plt.subplots(figsize=(10,5))
            data=[]; labels=[]
            for variant in ["BASE", f"FT_{args.tag}", f"HYB_ATTN_{args.tag}", f"HYB_MLP_{args.tag}", f"HYB_BOTH_{args.tag}"]:
                vals = sens_df.loc[sens_df["variant"]==variant, "length_std_across_instructions"].values
                if vals.size == 0: continue
                data.append(vals); labels.append(variant)
            if data:
                ax.boxplot(data, labels=labels, showfliers=False)
                ax.set_title("Instruction sensitivity distribution (per prompt std)")
                ax.set_ylabel("Std (tokens)")
                fig.tight_layout()
                fig.savefig(outdir / f"instruction_sensitivity_box_{args.tag}.png", dpi=170)
                plt.close(fig)
    except Exception as e:
        logging.warning("Instruction sensitivity plots failed: %s", e)

    # README
    lines = []
    lines.append("# Causal patching — outputs")
    lines.append("")
    lines.append("Feed these to AlpacaEval (or your evaluator):")
    lines.append("- `answers_BASE.json`")
    lines.append(f"- `answers_FT_{args.tag}.json`")
    lines.append(f"- `answers_HYB_ATTN_{args.tag}.json`  (BASE + FT attention @ layer {args.layer_index})")
    lines.append(f"- `answers_HYB_MLP_{args.tag}.json`   (BASE + FT MLP @ layer {args.layer_index})")
    lines.append(f"- `answers_HYB_BOTH_{args.tag}.json`  (BASE + FT attention+MLP @ layer {args.layer_index})")
    lines.append("")
    lines.append("Also included:")
    lines.append(f"- `generations_summary_{args.tag}.csv` (n_chars, ~n_tokens per generation)")
    lines.append("- Per-variant: `answer_length_hist_{VARIANT}.png`")
    lines.append("")
    lines.append("Comparative plots (all variants in the same figure, different colors):")
    lines.append(f"- `answer_length_hist_overlay_{args.tag}.png`")
    lines.append(f"- `answer_length_box_overlay_{args.tag}.png`")
    lines.append(f"- `answer_length_scatter_base_vs_variants_{args.tag}.png`")
    lines.append(f"- `length_closeness_to_FT_bars_{args.tag}.png`")
    lines.append(f"- `instruction_sensitivity_bars_{args.tag}.png`")
    lines.append(f"- `instruction_sensitivity_box_{args.tag}.png`")
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    log.info("All done. Artifacts at: %s", outdir)

if __name__ == "__main__":
    main()
