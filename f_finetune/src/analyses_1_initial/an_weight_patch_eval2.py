#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weight_patch_eval.py — Actual run with weight patching (answer-level robustness)

What it does
------------
• Loads BASE and FT models (HF names or local paths).
• Builds two hybrids from BASE by patching, at a chosen layer:
    - HYB_ATTN : replace only attention weights with FT's.
    - HYB_MLP  : replace only MLP weights with FT's.
• For each prompt_count:
    - Generate answers for instruction_original and its paraphrases
      with each of {BASE, FT, HYB_ATTN, HYB_MLP}.
    - Embed each answer via the model's last hidden layer (mean-pooled).
    - Compute D_model = mean cosine distance between that model’s original answer
      and its paraphrase answers (same prompt_count group).
• Report Δ = D_base − D_model (improvement vs BASE), per prompt and averaged.

Outputs
-------
[outdir]/
  generations.csv                      # all generated answers
  answer_distance_per_model.csv        # D_model per prompt_count & model
  answer_diff_of_diffs.csv             # pivot with D_*, Δ_* per prompt_count
  answer_diff_of_diffs_summary.png     # mean Δ bars (FT, HYB_ATTN, HYB_MLP)
  answer_diff_of_diffs_per_prompt.png  # grouped Δ bars per prompt_count
  report.md                            # numbers and file names in plain text

Usage
-----
python weight_patch_eval.py \
  --instructions_json /path/to/instructions.json \
  --base_model_name_or_path google/gemma-2-2b-it \
  --ft_model_name_or_path  /path/to/your/merged_ft_checkpoint \
  --layer_index 6 \
  --max_items 50 \
  --max_new_tokens 256 \
  --outdir ./weight_patch_outputs

Notes
-----
• Greedy decoding by default (deterministic). Add --do_sample for stochastic runs.
• You can filter paraphrase types with --keys or --keys_regex.
• If FT is a LoRA adapter, pass --ft_lora_adapter instead of --ft_model_name_or_path.
"""

from __future__ import annotations

import argparse, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# Utilities & data

def set_seed(seed=42):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

@dataclass
class Item:
    prompt_count: int
    instruction_original: str
    paraphrases: Dict[str, str]
    input_text: str

    def variants(self, include_original=True,
                 keys: Optional[List[str]]=None,
                 regex: Optional[str]=None) -> List[Tuple[str,str]]:
        out: List[Tuple[str,str]] = []
        if include_original:
            out.append(("instruction_original", self.instruction_original))
        for k, v in sorted(self.paraphrases.items()):
            if not k.startswith("instruct_"): 
                continue
            if keys and k not in keys: 
                continue
            if regex and re.search(regex, k) is None: 
                continue
            out.append((k, v))
        return out

def load_instruction_json(path: str | Path) -> List[Item]:
    data = json.loads(Path(path).read_text())
    items: List[Item] = []
    for obj in data:
        items.append(
            Item(
                prompt_count=int(obj["prompt_count"]),
                instruction_original=obj.get("instruction_original",""),
                paraphrases={k:v for k,v in obj.items() if k.startswith("instruct_")},
                input_text=(obj.get("input","") or "")
            )
        )
    return items

def build_prompt_text(instruction: str, input_text: str) -> str:
    return f"{instruction}\n\nInput: {input_text}" if (input_text and input_text.strip()) else instruction

def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """1 − cosine similarity, return as float."""
    a = a.view(-1).float()
    b = b.view(-1).float()
    return float(1.0 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


# Model access & weight patching

class BlockAccessor:
    """Robustly access blocks across Llama/Gemma/Mistral/OPT-like models."""
    def __init__(self, model: nn.Module, layer_index: int):
        if hasattr(model, "model") and hasattr(model.model, "layers"):        # Llama/Gemma2
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"): # GPT2/OPT-like
            layers = model.transformer.h
        else:
            raise TypeError("Unsupported model architecture (no .model.layers or .transformer.h).")
        self.block = layers[layer_index]
        self.attn = getattr(self.block, "self_attn", None) or getattr(self.block, "attention", None)
        self.mlp  = getattr(self.block, "mlp", None) or getattr(self.block, "feed_forward", None)
        if self.attn is None or self.mlp is None:
            raise TypeError("Could not access attention/MLP submodules.")
        # Attn
        self.q_proj = getattr(self.attn, "q_proj", None)
        self.k_proj = getattr(self.attn, "k_proj", None)
        self.v_proj = getattr(self.attn, "v_proj", None)
        self.o_proj = getattr(self.attn, "o_proj", None)
        # MLP
        self.up_proj   = getattr(self.mlp, "up_proj", None)
        self.gate_proj = getattr(self.mlp, "gate_proj", None)
        self.down_proj = getattr(self.mlp, "down_proj", None)

    def swap_attention_from(self, other: "BlockAccessor"):
        for name in ["q_proj","k_proj","v_proj","o_proj"]:
            src, dst = getattr(other, name, None), getattr(self, name, None)
            if src is not None and dst is not None:
                dst.weight.data.copy_(src.weight.data)
                if getattr(dst, "bias", None) is not None and getattr(src, "bias", None) is not None:
                    dst.bias.data.copy_(src.bias.data)

    def swap_mlp_from(self, other: "BlockAccessor"):
        for name in ["up_proj","gate_proj","down_proj"]:
            src, dst = getattr(other, name, None), getattr(self, name, None)
            if src is not None and dst is not None:
                dst.weight.data.copy_(src.weight.data)
                if getattr(dst, "bias", None) is not None and getattr(src, "bias", None) is not None:
                    dst.bias.data.copy_(src.bias.data)

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
        raise ValueError("Provide either --ft_model_name_or_path (merged FT) or --ft_lora_adapter (LoRA).")

    if ft_lora_adapter is not None:
        if not _HAS_PEFT:
            raise RuntimeError("peft not installed but --ft_lora_adapter was provided.")
        ft = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
        ft = PeftModel.from_pretrained(ft, ft_lora_adapter)
        ft = ft.merge_and_unload().eval()
    else:
        ft = AutoModelForCausalLM.from_pretrained(ft_model_name_or_path, torch_dtype=dtype).to(device).eval()

    return base, ft, tokenizer

def build_hybrid_models(base_model, ft_model, layer_index: int, device: str):
    import copy
    hyb_attn = copy.deepcopy(base_model).to(device)
    hyb_mlp  = copy.deepcopy(base_model).to(device)
    ft_blk   = BlockAccessor(ft_model, layer_index)
    ha_blk   = BlockAccessor(hyb_attn, layer_index)
    hm_blk   = BlockAccessor(hyb_mlp,  layer_index)
    ha_blk.swap_attention_from(ft_blk)
    hm_blk.swap_mlp_from(ft_blk)
    hyb_attn.eval(); hyb_mlp.eval()
    return hyb_attn, hyb_mlp


# Generation & embeddings

def generate_answer(model, tokenizer, text: str, device: str,
                    max_new_tokens: int, do_sample: bool,
                    temperature: float, top_p: float, top_k: int) -> str:
    enc = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0][enc["input_ids"].shape[1]:]  # only new tokens
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def text_embedding(model, tokenizer, text: str, device: str) -> torch.Tensor:
    """Mean-pooled last-layer hidden state for the answer text."""
    if not text:
        text = "."
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True, return_dict=True)
        last = outputs.hidden_states[-1][0]  # [T, D]
    return last.float().mean(dim=0)          # [D] (use float for stable cos)


# Evaluation loop

def run_eval(
    items: List[Item],
    base, ft, tokenizer,
    layer_index: int,
    device: str,
    outdir: Path,
    keys: Optional[List[str]],
    keys_regex: Optional[str],
    max_items: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
):
    # Build hybrids
    hyb_attn, hyb_mlp = build_hybrid_models(base, ft, layer_index, device)
    MODELS = {"BASE": base, "FT": ft, "HYB_ATTN": hyb_attn, "HYB_MLP": hyb_mlp}

    # For logging
    decode_cfg = f"max_new_tokens={max_new_tokens}, do_sample={do_sample}, temperature={temperature}, top_p={top_p}, top_k={top_k}"
    (outdir / "decode_config.txt").write_text(decode_cfg, encoding="utf-8")

    rows_gen = []
    rows_metrics = []

    processed = 0
    for item in items:
        if processed >= max_items:
            break
        variants = item.variants(include_original=True, keys=keys, regex=keys_regex)
        if len(variants) < 2:   # need at least original + one paraphrase
            continue

        # Generate once per model for original prompt
        orig_answers: Dict[str, str] = {}
        orig_embeds: Dict[str, torch.Tensor] = {}

        orig_prompt = build_prompt_text(item.instruction_original, item.input_text)
        for mname, model in MODELS.items():
            ans = generate_answer(model, tokenizer, orig_prompt, device,
                                  max_new_tokens, do_sample, temperature, top_p, top_k)
            emb = text_embedding(model, tokenizer, ans, device)
            orig_answers[mname] = ans
            orig_embeds[mname]  = emb
            rows_gen.append({
                "prompt_count": item.prompt_count,
                "key": "instruction_original",
                "model": mname,
                "answer": ans
            })

        # Generate paraphrases
        for key, phr in variants:
            if key == "instruction_original":
                continue
            ptxt = build_prompt_text(phr, item.input_text)
            for mname, model in MODELS.items():
                ans = generate_answer(model, tokenizer, ptxt, device,
                                      max_new_tokens, do_sample, temperature, top_p, top_k)
                rows_gen.append({
                    "prompt_count": item.prompt_count,
                    "key": key,
                    "model": mname,
                    "answer": ans
                })

        # Compute D_model per model (distance: original answer vs paraphrase answers)
        df_this = pd.DataFrame([r for r in rows_gen if r["prompt_count"] == item.prompt_count])
        for mname in MODELS.keys():
            emb_orig = orig_embeds[mname]
            dlist = []
            for _, r in df_this[(df_this["model"] == mname) & (df_this["key"] != "instruction_original")].iterrows():
                emb_para = text_embedding(MODELS[mname], tokenizer, r["answer"], device)
                dlist.append(cosine_distance(emb_orig, emb_para))
            D_model = float(np.mean(dlist)) if dlist else np.nan
            rows_metrics.append({
                "prompt_count": item.prompt_count,
                "model": mname,
                "D_model_answer": D_model,
                "N_paraphrases": len(dlist)
            })

        processed += 1

    # Save generations
    df_gen = pd.DataFrame(rows_gen)
    df_gen.to_csv(outdir / "generations.csv", index=False)

    # Save per-model distances
    df_metrics = pd.DataFrame(rows_metrics)
    df_metrics.to_csv(outdir / "answer_distance_per_model.csv", index=False)

    # Pivot and compute deltas
    pivot = df_metrics.pivot(index="prompt_count", columns="model", values="D_model_answer").reset_index()
    for col in ["BASE","FT","HYB_ATTN","HYB_MLP"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["DELTA_FT"]       = pivot["BASE"] - pivot["FT"]
    pivot["DELTA_HYB_ATTN"] = pivot["BASE"] - pivot["HYB_ATTN"]
    pivot["DELTA_HYB_MLP"]  = pivot["BASE"] - pivot["HYB_MLP"]
    pivot.to_csv(outdir / "answer_diff_of_diffs.csv", index=False)

    # Summary plots
    # Mean Δ bars
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["FT","HYB_ATTN","HYB_MLP"]
    means  = [pd.to_numeric(pivot[f"DELTA_{k}"], errors="coerce").mean() for k in labels]
    ax.bar(labels, means)
    ax.set_title(f"Answer-level robustness Δ vs BASE (layer {layer_index})")
    ax.set_ylabel("Δ = D_base - D_model (higher is better)")
    fig.tight_layout()
    fig.savefig(outdir / "answer_diff_of_diffs_summary.png", dpi=180)
    plt.close(fig)

    # Per-prompt grouped bars
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(pivot.shape[0])
    width = 0.25
    ax.bar(x - width, pd.to_numeric(pivot["DELTA_FT"], errors="coerce"), width, label="FT")
    ax.bar(x,         pd.to_numeric(pivot["DELTA_HYB_ATTN"], errors="coerce"), width, label="HYB_ATTN")
    ax.bar(x + width, pd.to_numeric(pivot["DELTA_HYB_MLP"], errors="coerce"), width, label="HYB_MLP")
    ax.set_xticks(x); ax.set_xticklabels([str(int(i)) for i in pivot["prompt_count"].fillna(-1)])
    ax.set_xlabel("prompt_count"); ax.set_ylabel("Δ (higher is better)")
    ax.set_title(f"Answer-level Δ per prompt (layer {layer_index})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "answer_diff_of_diffs_per_prompt.png", dpi=200)
    plt.close(fig)

    # Report with actual numbers and file names
    report_lines = []
    report_lines.append(f"# Weight patching analysis — layer {layer_index}\n")
    report_lines.append("## Files\n")
    report_lines.append("- `generations.csv`")
    report_lines.append("- `answer_distance_per_model.csv`")
    report_lines.append("- `answer_diff_of_diffs.csv`")
    report_lines.append("- `answer_diff_of_diffs_summary.png`")
    report_lines.append("- `answer_diff_of_diffs_per_prompt.png`\n")
    report_lines.append("## Mean Δ (D_base − D_model)\n")
    for k, m in zip(labels, means):
        report_lines.append(f"- {k}: **{float(m):.4f}**")
    report_lines.append("")
    report_lines.append("Δ > 0 means the model (FT or hybrid) is **more robust** than BASE.")
    (outdir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    # Also print succinct summary
    print("Done.")
    print(f"Saved CSVs and figures under: {outdir}")
    print("Mean Δ (D_base − D_model):")
    for k, m in zip(labels, means):
        print(f"  {k:9s}: {m:.4f}")


# CLI

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Weight patching eval (answer-level robustness)")
    ap.add_argument("--instructions_json", type=str, required=True)

    ap.add_argument("--base_model_name_or_path", type=str, required=True)
    ap.add_argument("--ft_model_name_or_path", type=str, default=None)
    ap.add_argument("--ft_lora_adapter", type=str, default=None)

    ap.add_argument("--layer_index", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)

    # decoding controls
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)

    # data controls
    ap.add_argument("--max_items", type=int, default=50, help="Max prompt_count items to process")
    ap.add_argument("--keys", type=str, default=None,
                    help="Comma-separated paraphrase keys to include (e.g., 'instruct_polite_request,instruct_one_typo_punctuation')")
    ap.add_argument("--keys_regex", type=str, default=None,
                    help="Regex to include paraphrase keys (e.g., '.*polite.*|.*sardonic.*')")

    args = ap.parse_args()
    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    items = load_instruction_json(args.instructions_json)
    keys = [k.strip() for k in args.keys.split(",")] if args.keys else None

    base, ft, tokenizer = build_model_and_tokenizer(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, args.device
    )

    run_eval(
        items=items,
        base=base, ft=ft, tokenizer=tokenizer,
        layer_index=args.layer_index,
        device=args.device,
        outdir=outdir,
        keys=keys,
        keys_regex=args.keys_regex,
        max_items=args.max_items,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
