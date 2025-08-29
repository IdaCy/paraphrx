#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_script08.py  — readable outputs, expert-complete

Adds:
- Averaged-over-H Jacobian probe (as expert requested, not just at the mean point)
- Robust diff-of-diffs summary plot (handles NaNs; annotates N)
- Attention filtering via --attn_keys and/or --attn_keys_regex
- PCA 2D scatter (no unreadable raw vectors dump)
- Clear per-item & global CSV summaries
- Markdown report with narrative conclusions (report_layer{L}.md)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
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
from matplotlib import gridspec

from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# ------------------------------
# Utils
# ------------------------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return (a * b).sum(dim=-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        m = mask.to(device=x.device, dtype=x.dtype)
        return (x * m.unsqueeze(-1)).sum(dim=0) / (m.sum() + 1e-8)
    elif x.dim() == 3:
        m = mask.to(device=x.device, dtype=x.dtype)
        return (x * m.unsqueeze(-1)).sum(dim=1) / (m.sum(dim=1, keepdim=True) + 1e-8)
    else:
        raise ValueError("masked_mean expects [T,D] or [B,T,D] tensors")


# ------------------------------
# Data
# ------------------------------

@dataclass
class Item:
    prompt_count: int
    instruction_original: str
    paraphrases: Dict[str, str]
    input_text: str

    def all_prompt_variants(self, include_original: bool = True) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        if include_original:
            pairs.append(("instruction_original", self.instruction_original))
        for k, v in sorted(self.paraphrases.items()):
            if k.startswith("instruct_"):
                pairs.append((k, v))
        return pairs


def load_instruction_json(path: str | Path) -> List[Item]:
    data = json.loads(Path(path).read_text())
    items: List[Item] = []
    for obj in data:
        pc = int(obj["prompt_count"])
        inp = obj.get("input", "") or ""
        instr = obj.get("instruction_original", "")
        paraphrases = {k: v for k, v in obj.items() if k.startswith("instruct_")}
        items.append(Item(prompt_count=pc,
                          instruction_original=instr,
                          paraphrases=paraphrases,
                          input_text=inp))
    return items


# ------------------------------
# Tokenization
# ------------------------------

@dataclass
class Encoded:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_mask: torch.Tensor


def build_prompt_text(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return f"{instruction}\n\nInput: {input_text}"
    return instruction


def encode_prompt(tokenizer, text: str, device: str, prompt_span: str = "no_bos") -> Encoded:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"][0]
    attn_mask = enc["attention_mask"][0]
    T = input_ids.shape[0]

    prompt_mask = torch.ones(T, dtype=torch.bool)
    if prompt_span == "no_bos" and T > 0:
        prompt_mask[0] = False
    elif prompt_span == "pre_eos":
        eos = (input_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos) > 0:
            prompt_mask[eos[0]:] = False

    return Encoded(input_ids=input_ids.to(device),
                   attention_mask=attn_mask.to(device),
                   prompt_mask=prompt_mask.to(device))


# ------------------------------
# Model access & hooks
# ------------------------------

class BlockAccessor:
    def __init__(self, model: nn.Module, layer_index: int):
        self.model = model
        self.layer_index = layer_index
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.layers = model.transformer.h
        else:
            raise TypeError("Unsupported model architecture.")
        self.block = self.layers[layer_index]
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
        self.up_proj = getattr(self.mlp, "up_proj", None)
        self.gate_proj = getattr(self.mlp, "gate_proj", None)
        self.down_proj = getattr(self.mlp, "down_proj", None)

    def swap_attention_from(self, other: "BlockAccessor"):
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            src = getattr(other, name, None); dst = getattr(self, name, None)
            if src is not None and dst is not None:
                dst.weight.data.copy_(src.weight.data)
                if hasattr(dst, "bias") and hasattr(src, "bias") and dst.bias is not None and src.bias is not None:
                    dst.bias.data.copy_(src.bias.data)

    def swap_mlp_from(self, other: "BlockAccessor"):
        for name in ["up_proj", "gate_proj", "down_proj"]:
            src = getattr(other, name, None); dst = getattr(self, name, None)
            if src is not None and dst is not None:
                dst.weight.data.copy_(src.weight.data)
                if hasattr(dst, "bias") and hasattr(src, "bias") and dst.bias is not None and src.bias is not None:
                    dst.bias.data.copy_(src.bias.data)


class HookHandles:
    def __init__(self):
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
    def add(self, h): self.handles.append(h)
    def remove_all(self):
        for h in self.handles:
            try: h.remove()
            except Exception: pass
        self.handles.clear()


@dataclass
class CapturedActivations:
    resid_pre: List[torch.Tensor]
    resid_post: List[torch.Tensor]
    attn_probs: List[torch.Tensor]


def capture_layer_activations(model, layer_index: int, device: str):
    layers = BlockAccessor(model, layer_index).layers
    block = layers[layer_index]
    attn = getattr(block, "self_attn", None) or getattr(block, "attention", None)

    resid_pre_list: List[torch.Tensor] = []
    resid_post_list: List[torch.Tensor] = []
    attn_probs_list: List[torch.Tensor] = []

    hooks = HookHandles()

    def pre_hook(module, inputs):
        resid_pre_list.append(inputs[0].detach().to("cpu"))
    def block_forward_hook(module, inputs, outputs):
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        resid_post_list.append(hidden_states.detach().to("cpu"))
    def attn_forward_hook(module, inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) >= 2 and outputs[1] is not None:
            attn_probs_list.append(outputs[1].detach().to("cpu"))  # [B,H,T,T]

    hooks.add(block.register_forward_pre_hook(pre_hook))
    hooks.add(block.register_forward_hook(block_forward_hook))
    if attn is not None:
        hooks.add(attn.register_forward_hook(attn_forward_hook))

    def run(encoded_batch: List[Encoded]) -> CapturedActivations:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [e.input_ids for e in encoded_batch], batch_first=True, padding_value=model.config.pad_token_id or 0
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [e.attention_mask for e in encoded_batch], batch_first=True, padding_value=0
        )
        with torch.no_grad():
            _ = model(input_ids=input_ids.to(device),
                      attention_mask=attention_mask.to(device),
                      output_attentions=True, return_dict=True)
        attns = []
        for tensor in attn_probs_list:
            for b in range(tensor.shape[0]):
                attns.append(tensor[b])
        return CapturedActivations(resid_pre=resid_pre_list[:],
                                   resid_post=resid_post_list[:],
                                   attn_probs=attns)

    run.hooks = hooks
    return run


# ------------------------------
# Jacobian probe
# ------------------------------

def mlp_function_from_block(model, layer_index: int):
    block = BlockAccessor(model, layer_index).layers[layer_index]
    mlp = getattr(block, "mlp", None) or getattr(block, "feed_forward", None)
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    def f(h: torch.Tensor) -> torch.Tensor:
        mlp.eval()
        with torch.no_grad():
            h2 = h.to(device=model_device, dtype=model_dtype)
            return mlp(h2)
    return f


def jacobian_directional_norms(mlp_f, h: torch.Tensor, directions: torch.Tensor, eps: float) -> torch.Tensor:
    directions = directions.to(h.dtype).to(h.device)
    K, D = directions.shape
    with torch.no_grad():
        diffs = []
        for k in range(K):
            d = directions[k]
            y_pos = mlp_f(h + eps * d)
            y_neg = mlp_f(h - eps * d)
            g = (y_pos - y_neg) / (2.0 * eps)
            diffs.append(g.norm(p=2))
        return torch.stack(diffs)


# ------------------------------
# Representations & distances
# ------------------------------

def prompt_representation_from_layer(model, layer_index: int, enc: Encoded) -> torch.Tensor:
    runner = capture_layer_activations(model, layer_index, next(model.parameters()).device)
    try:
        caps = runner([enc])
        if not caps.resid_pre:
            raise RuntimeError("Failed to capture resid_pre activations.")
        pre = caps.resid_pre[0][0]
        rep = masked_mean(pre, enc.prompt_mask)
        return rep
    finally:
        runner.hooks.remove_all()


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1.0 - cosine(a.unsqueeze(0), b.unsqueeze(0)).item())


# ------------------------------
# Models
# ------------------------------

def build_model_and_tokenizer(
    base_model_name_or_path: str,
    ft_model_name_or_path: Optional[str],
    ft_lora_adapter: Optional[str],
    device: str,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
    ).to(device)
    base.eval()

    if ft_model_name_or_path is None and ft_lora_adapter is None:
        raise ValueError("Provide either --ft_model_name_or_path (merged FT) or --ft_lora_adapter (LoRA).")

    if ft_lora_adapter is not None:
        if not _HAS_PEFT:
            raise RuntimeError("peft not installed but --ft_lora_adapter provided.")
        ft = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
        ).to(device)
        ft = PeftModel.from_pretrained(ft, ft_lora_adapter)
        ft = ft.merge_and_unload()
        ft.eval()
    else:
        ft = AutoModelForCausalLM.from_pretrained(
            ft_model_name_or_path, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
        ).to(device)
        ft.eval()

    return base, ft, tokenizer


def build_hybrid_models(base_model, ft_model, layer_index: int, device: str):
    import copy
    hybrid_attn = copy.deepcopy(base_model).to(device)
    hybrid_mlp  = copy.deepcopy(base_model).to(device)
    ft_blk = BlockAccessor(ft_model, layer_index)
    ha_blk = BlockAccessor(hybrid_attn, layer_index)
    hm_blk = BlockAccessor(hybrid_mlp,  layer_index)
    ha_blk.swap_attention_from(ft_blk)
    hm_blk.swap_mlp_from(ft_blk)
    for m in [hybrid_attn, hybrid_mlp]:
        m.eval()
    return hybrid_attn, hybrid_mlp


# ------------------------------
# 1) Diff-of-diffs (component swaps)
# ------------------------------

def diff_of_diffs_for_item(
    models: Dict[str, nn.Module],
    tokenizer,
    item: Item,
    layer_index: int,
    device: str,
    prompt_span: str,
) -> Dict[str, float]:
    orig_text = build_prompt_text(item.instruction_original, item.input_text)
    enc_orig = encode_prompt(tokenizer, orig_text, device, prompt_span=prompt_span)

    pps = [(k, v) for k, v in item.paraphrases.items() if k.startswith("instruct_")]
    if len(pps) == 0:
        return {}

    reps_orig = {name: prompt_representation_from_layer(model, layer_index, enc_orig)
                 for name, model in models.items()}

    dists = {name: [] for name in models.keys()}
    for key, para in pps:
        enc = encode_prompt(tokenizer, build_prompt_text(para, item.input_text), device, prompt_span=prompt_span)
        for name, model in models.items():
            rep = prompt_representation_from_layer(model, layer_index, enc)
            dists[name].append(cosine_distance(rep, reps_orig[name]))

    avgD = {name: float(np.nanmean(vals)) if len(vals) else np.nan for name, vals in dists.items()}
    baseD = avgD.get("BASE", np.nan)
    return {name: (baseD - v) if (not np.isnan(baseD) and not np.isnan(v)) else np.nan
            for name, v in avgD.items()}


def run_diff_of_diffs(
    items: List[Item],
    base: nn.Module,
    ft: nn.Module,
    tokenizer,
    layer_index: int,
    device: str,
    outdir: Path,
    prompt_span: str,
) -> pd.DataFrame:
    hybrid_attn, hybrid_mlp = build_hybrid_models(base, ft, layer_index, device)
    models = {"BASE": base, "FT": ft, "HYB_ATTN": hybrid_attn, "HYB_MLP": hybrid_mlp}

    rows = []
    for item in items:
        row = {"prompt_count": item.prompt_count}
        row.update(diff_of_diffs_for_item(models, tokenizer, item, layer_index, device, prompt_span))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("prompt_count")
    csv_path = outdir / f"diff_of_diffs_layer{layer_index}.csv"
    df.to_csv(csv_path, index=False)

    # Robust summary plot
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = []
    labels = ["FT", "HYB_ATTN", "HYB_MLP"]
    Ns = []
    for col in labels:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        Ns.append(int(series.shape[0]))
        bars.append(series.mean() if series.shape[0] else 0.0)
    ax.bar(labels, bars)
    ax.set_title(f"Diff-of-diffs (avg across items) — layer {layer_index}\nN={Ns}")
    ax.set_ylabel("Δ = D_base - D_model (cosine distance)")
    fig.tight_layout()
    png_summary = outdir / f"diff_of_diffs_layer{layer_index}_summary.png"
    fig.savefig(png_summary, dpi=180)
    plt.close(fig)

    return df


# ------------------------------
# 2) Paraphrase subspace shrinkage (PCA + Jacobian, averaged over H)
# ------------------------------

def paraphrase_pca_and_jacobian(
    items: List[Item],
    model: nn.Module,
    tokenizer,
    layer_index: int,
    device: str,
    outdir: Path,
    prompt_span: str,
    topk: int = 8,
    eps: float = 1e-3,
    scatter_labels: bool = False,
    max_prompts_for_overview: int = 50,
) -> pd.DataFrame:
    f_mlp = mlp_function_from_block(model, layer_index)

    summary_rows = []
    all_scatter = []   # list of (pc1, pc2, prompt_id)
    all_scree = []     # list of (prompt_id, x (1..m), y(varexpl))
    for item in items:
        H_list = []
        keys = []
        for key, para in item.all_prompt_variants(include_original=True):
            if not key.startswith("instruct_") and key != "instruction_original":
                continue
            enc = encode_prompt(tokenizer, build_prompt_text(para, item.input_text), device, prompt_span=prompt_span)
            h = prompt_representation_from_layer(model, layer_index, enc)  # [D]
            H_list.append(h.cpu().numpy().astype(np.float32))
            keys.append(key)
        if len(H_list) < 3:
            continue
        H = np.stack(H_list, axis=0)  # [N, D]
        N, D = H.shape

        # PCA on centered data
        H_mean = H.mean(axis=0, keepdims=True)
        Hc = H - H_mean
        U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
        var = (S ** 2)
        varexpl = var / var.sum() if var.sum() > 0 else np.zeros_like(var)

        # Scree plot (per-prompt)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(np.arange(1, len(S) + 1), varexpl, marker="o")
        ax.set_title(f"PCA scree — prompt {item.prompt_count} (layer {layer_index})")
        ax.set_xlabel("component")
        ax.set_ylabel("variance explained")
        fig.tight_layout()
        fig.savefig(outdir / f"pca_scree_prompt{item.prompt_count}_layer{layer_index}.png", dpi=180)
        plt.close(fig)

        # Collect for ALL-in-one scree (truncate if many)
        if len(all_scree) < max_prompts_for_overview:
            for i, ve in enumerate(varexpl, start=1):
                all_scree.append((item.prompt_count, i, float(ve)))

        # 2D scatter in PC space (per-prompt)
        PCs = Vt[:2]  # [2, D]
        proj = Hc @ PCs.T  # [N, 2]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(proj[:,0], proj[:,1])
        if scatter_labels:
            for i, key in enumerate(keys):
                ax.text(proj[i,0], proj[i,1], key, fontsize=7, alpha=0.8)
        ax.set_title(f"Paraphrase PCA scatter — prompt {item.prompt_count} (layer {layer_index})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.tight_layout()
        fig.savefig(outdir / f"pca_scatter_prompt{item.prompt_count}_layer{layer_index}.png", dpi=180)
        plt.close(fig)

        # Collect for ALL-in-one scatter (truncate if many)
        if len(all_scatter) < max_prompts_for_overview:
            for i in range(proj.shape[0]):
                all_scatter.append((float(proj[i,0]), float(proj[i,1]), item.prompt_count))

        # Build direction sets
        k = int(min(topk, Vt.shape[0]))
        PCs_k = Vt[:k]  # [k,D]
        model_dtype = next(model.parameters()).dtype
        PCs_t = torch.from_numpy(PCs_k).to(device).to(model_dtype)

        # Average directional norms across ALL h_i in H
        h_tensor = torch.from_numpy(H).to(device).to(model_dtype)  # [N,D]
        h_mean_t = torch.from_numpy(H_mean.squeeze(0)).to(device).to(model_dtype)
        PCs_t = F.normalize(PCs_t, dim=-1)
        rand_dirs = torch.randn_like(PCs_t); rand_dirs = F.normalize(rand_dirs, dim=-1)
        mean_dir = F.normalize(h_mean_t, dim=-1, eps=1e-8).unsqueeze(0)

        norms_pcs_list, norms_rand_list, norms_mean_list = [], [], []
        for i in range(h_tensor.shape[0]):
            hi = h_tensor[i]
            norms_pcs_list.append(jacobian_directional_norms(f_mlp, hi, PCs_t, eps=eps).cpu().numpy())
            norms_rand_list.append(jacobian_directional_norms(f_mlp, hi, rand_dirs, eps=eps).cpu().numpy())
            norms_mean_list.append(jacobian_directional_norms(f_mlp, hi, mean_dir, eps=eps).cpu().numpy())
        norms_pcs = np.stack(norms_pcs_list, axis=0).mean(axis=0)   # [k]
        norms_rand = np.stack(norms_rand_list, axis=0).mean(axis=0) # [k]
        norms_mean = np.stack(norms_mean_list, axis=0).mean(axis=0) # [1]

        # Per-prompt bar plot
        fig, ax = plt.subplots(figsize=(6, 4))
        means = [norms_pcs.mean(), norms_rand.mean(), norms_mean.mean()]
        ax.bar(["PC", "RANDOM", "MEAN"], means)
        ax.set_title(f"MLP directional Jacobian norms — prompt {item.prompt_count} (layer {layer_index})")
        ax.set_ylabel("‖(f(h+εd)-f(h-εd))‖ / (2ε) (averaged over H)")
        fig.tight_layout()
        fig.savefig(outdir / f"jacobian_norms_prompt{item.prompt_count}_layer{layer_index}.png", dpi=180)
        plt.close(fig)

        # Summary row
        summary_rows.append({
            "prompt_count": item.prompt_count,
            "N_paraphrases": int(N),
            "k_top": int(k),
            "pc1_var_expl": float(varexpl[0]) if len(varexpl) else np.nan,
            "cum_var_expl_topk": float(varexpl[:k].sum()) if len(varexpl) else np.nan,
            "jac_PC_mean": float(means[0]),
            "jac_RANDOM_mean": float(means[1]),
            "jac_MEAN": float(means[2]),
            "contraction_ratio_PC_over_RANDOM": float(means[0] / means[1]) if means[1] != 0 else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("prompt_count")
    summary_csv = outdir / f"paraphrase_subspace_jacobian_layer{layer_index}.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Global plot: mean of means
    if not summary_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["PC", "RANDOM", "MEAN"],
               [summary_df["jac_PC_mean"].mean(),
                summary_df["jac_RANDOM_mean"].mean(),
                summary_df["jac_MEAN"].mean()])
        ax.set_title(f"Avg Jacobian norms across items — layer {layer_index}")
        ax.set_ylabel("mean directional norm (averaged over items)")
        fig.tight_layout()
        fig.savefig(outdir / f"jacobian_norms_layer{layer_index}_summary.png", dpi=180)
        plt.close(fig)

        # NEW: All-prompts overview where each prompt is a colored bar triple
        fig, ax = plt.subplots(figsize=(10, 6))
        prompts = summary_df["prompt_count"].tolist()
        x = np.arange(3)  # PC, RANDOM, MEAN
        width = 0.8 / max(len(prompts), 1)
        for i, pcnt in enumerate(prompts):
            vals = [summary_df.loc[summary_df["prompt_count"]==pcnt, "jac_PC_mean"].values[0],
                    summary_df.loc[summary_df["prompt_count"]==pcnt, "jac_RANDOM_mean"].values[0],
                    summary_df.loc[summary_df["prompt_count"]==pcnt, "jac_MEAN"].values[0]]
            ax.bar(x + i*width - 0.4, vals, width=width, label=f"prompt {pcnt}")
        ax.set_xticks(x)
        ax.set_xticklabels(["PC","RANDOM","MEAN"])
        ax.set_title(f"Jacobian norms per prompt — layer {layer_index}")
        ax.set_ylabel("‖(f(h+εd)-f(h-εd))‖ / (2ε)")
        if len(prompts) <= 20:
            ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"jacobian_norms_layer{layer_index}_ALLPROMPTS.png", dpi=200)
        plt.close(fig)

    # NEW: ALL-in-one PCA scatter (overview)
    if len(all_scatter) > 0:
        scatter_df = pd.DataFrame(all_scatter, columns=["pc1","pc2","prompt_count"])
        fig, ax = plt.subplots(figsize=(8,6))
        for pcnt, sub in scatter_df.groupby("prompt_count"):
            ax.scatter(sub["pc1"], sub["pc2"], label=f"prompt {pcnt}", s=15)
        ax.set_title(f"Paraphrase PCA scatter — ALL prompts (layer {layer_index})")
        ax.set_xlabel("PC1 (per-prompt frame)")
        ax.set_ylabel("PC2 (per-prompt frame)")
        if scatter_df["prompt_count"].nunique() <= 20:
            ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"pca_scatter_layer{layer_index}_ALL.png", dpi=200)
        plt.close(fig)

    # NEW: ALL-in-one scree (overview)
    if len(all_scree) > 0:
        scree_df = pd.DataFrame(all_scree, columns=["prompt_count","component","variance_explained"])
        fig, ax = plt.subplots(figsize=(8,6))
        for pcnt, sub in scree_df.groupby("prompt_count"):
            ax.plot(sub["component"], sub["variance_explained"], marker="o", label=f"prompt {pcnt}")
        ax.set_title(f"PCA scree — ALL prompts (layer {layer_index})")
        ax.set_xlabel("component")
        ax.set_ylabel("variance explained")
        if scree_df["prompt_count"].nunique() <= 20:
            ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"pca_scree_layer{layer_index}_ALL.png", dpi=200)
        plt.close(fig)

    return summary_df


# ------------------------------
# 3) Attention inspection with filtering (grid output per paraphrase)
# ------------------------------

def attention_heatmaps(
    items: List[Item],
    model: nn.Module,
    tokenizer,
    layer_index: int,
    device: str,
    outdir: Path,
    prompt_span: str,
    max_items: int = 25,
    attn_keys: Optional[List[str]] = None,
    attn_keys_regex: Optional[str] = None,
):
    rgx = re.compile(attn_keys_regex) if attn_keys_regex else None

    count = 0
    for item in items:
        for key, para in item.all_prompt_variants(include_original=True):
            if attn_keys is not None and len(attn_keys) and key not in attn_keys:
                continue
            if rgx is not None and not rgx.search(key):
                continue
            if count >= max_items:
                break

            enc = encode_prompt(tokenizer, build_prompt_text(para, item.input_text), device, prompt_span=prompt_span)
            runner = capture_layer_activations(model, layer_index, device)
            try:
                caps = runner([enc])
            finally:
                runner.hooks.remove_all()

            if not caps.attn_probs:
                continue

            att = caps.attn_probs[0]  # [H, T, T]
            H, T, _ = att.shape
            avg = att.mean(0).numpy()

            # --- grid: first tile is avg, then one per head ---
            nplots = H + 1
            cols = int(np.ceil(np.sqrt(nplots)))
            rows = int(np.ceil(nplots / cols))

            fig = plt.figure(figsize=(3.2*cols, 3.2*rows))
            gs = gridspec.GridSpec(rows, cols, figure=fig)

            # avg at (0)
            ax = fig.add_subplot(gs[0, 0])
            im = ax.imshow(avg, aspect="auto", origin="lower", interpolation="nearest")
            ax.set_title("avg-head")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # heads
            idx = 1
            for h in range(H):
                r = idx // cols
                c = idx % cols
                ax = fig.add_subplot(gs[r, c])
                im = ax.imshow(att[h].numpy(), aspect="auto", origin="lower", interpolation="nearest")
                ax.set_title(f"head {h}")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                idx += 1

            fig.suptitle(f"Layer {layer_index} attention grid — {key} (prompt {item.prompt_count})", y=0.995)
            fig.tight_layout()
            fig.savefig(outdir / f"attn_grid_layer{layer_index}_{item.prompt_count}_{key}.png", dpi=170)
            plt.close(fig)

            count += 1
        if count >= max_items:
            break


# ------------------------------
# Markdown report (with actual file names)
# ------------------------------

def write_markdown_report(outdir: Path, layer_index: int,
                          dod_df: Optional[pd.DataFrame],
                          subspace_df: Optional[pd.DataFrame]):
    md = []
    def add(s=""): md.append(s)

    add(f"# Paraphrase-robustness analysis — layer {layer_index}")
    add("This report summarizes the three expert-requested analyses and lists the generated files.")
    add("")

    # Diff-of-diffs section
    add("## 1) Causal component replacement (diff-of-diffs)")
    dod_csv = outdir / f"diff_of_diffs_layer{layer_index}.csv"
    dod_png = outdir / f"diff_of_diffs_layer{layer_index}_summary.png"
    if dod_csv.exists():
        add(f"- CSV: `{dod_csv.name}`")
    if dod_png.exists():
        add(f"- Summary figure: `{dod_png.name}`")
        add(f"![diff-of-diffs summary]({dod_png.name})")
    if dod_df is not None and not dod_df.empty:
        means = dod_df[["FT","HYB_ATTN","HYB_MLP"]].apply(pd.to_numeric, errors="coerce").mean(skipna=True)
        Ns = dod_df[["FT","HYB_ATTN","HYB_MLP"]].apply(pd.to_numeric, errors="coerce").count()
        add(f"- Mean Δ across items: FT `{means.get('FT', np.nan):.4f}` (N={int(Ns.get('FT',0))}), "
            f"HYB_ATTN `{means.get('HYB_ATTN', np.nan):.4f}` (N={int(Ns.get('HYB_ATTN',0))}), "
            f"HYB_MLP `{means.get('HYB_MLP', np.nan):.4f}` (N={int(Ns.get('HYB_MLP',0))})")
    add("")

    # Subspace shrinkage
    add("## 2) Paraphrase-subspace shrinkage (PCA + Jacobian)")
    jac_csv = outdir / f"paraphrase_subspace_jacobian_layer{layer_index}.csv"
    jac_png = outdir / f"jacobian_norms_layer{layer_index}_summary.png"
    jac_all = outdir / f"jacobian_norms_layer{layer_index}_ALLPROMPTS.png"
    scat_all = outdir / f"pca_scatter_layer{layer_index}_ALL.png"
    scree_all = outdir / f"pca_scree_layer{layer_index}_ALL.png"

    files = []
    if jac_csv.exists(): files.append(jac_csv.name)
    if jac_png.exists(): files.append(jac_png.name)
    if jac_all.exists(): files.append(jac_all.name)
    if scat_all.exists(): files.append(scat_all.name)
    if scree_all.exists(): files.append(scree_all.name)
    if files:
        add("- Files:")
        for f in files:
            add(f"  - `{f}`")

    if subspace_df is not None and not subspace_df.empty:
        add(f"- Avg contraction ratio PC/RANDOM: `{subspace_df['contraction_ratio_PC_over_RANDOM'].mean():.3f}`")
        add(f"- Avg MEAN sensitivity: `{subspace_df['jac_MEAN'].mean():.4f}`")
        add(f"- Avg PC1 variance explained: `{subspace_df['pc1_var_expl'].mean():.3f}`; "
            f"Avg cumulative top-k: `{subspace_df['cum_var_expl_topk'].mean():.3f}`")

    # List per-prompt figures (scree, scatter, jacobian)
    add("")
    add("- Per-prompt figures:")
    per_prompt_imgs = sorted(list(outdir.glob(f"pca_scree_prompt*_layer{layer_index}.png"))) + \
                      sorted(list(outdir.glob(f"pca_scatter_prompt*_layer{layer_index}.png"))) + \
                      sorted(list(outdir.glob(f"jacobian_norms_prompt*_layer{layer_index}.png")))
    if per_prompt_imgs:
        for p in per_prompt_imgs:
            add(f"  - `{p.name}`")

    add("")
    add("## 3) Attention patterns")
    grids = sorted(list(outdir.glob(f"attn_grid_layer{layer_index}_*.png")))
    if grids:
        add("- Grid images (avg + all heads in one file per paraphrase):")
        for g in grids:
            add(f"  - `{g.name}`")

    (outdir / f"report_layer{layer_index}.md").write_text("\n".join(md), encoding="utf-8")


# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paraphrase robustness analyses (expert-guided, readable outputs)")

    parser.add_argument("--instructions_json", type=str, required=True, help="Path to instructions JSON")
    parser.add_argument("--answers_json", type=str, default=None)
    parser.add_argument("--scores_json", type=str, default=None)

    parser.add_argument("--base_model_name_or_path", type=str, required=True)
    parser.add_argument("--ft_model_name_or_path", type=str, default=None)
    parser.add_argument("--ft_lora_adapter", type=str, default=None)

    parser.add_argument("--layer_index", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--prompt_span", type=str, default="no_bos", choices=["no_bos", "all", "pre_eos"])

    parser.add_argument("--run_diff_of_diffs", action="store_true")
    parser.add_argument("--run_paraphrase_subspace", action="store_true")
    parser.add_argument("--run_attention_inspection", action="store_true")

    parser.add_argument("--topk_pca", type=int, default=8)
    parser.add_argument("--eps_jacobian", type=float, default=1e-3)
    parser.add_argument("--scatter_labels", action="store_true")
    parser.add_argument("--max_prompts_for_overview", type=int, default=50,
                        help="Max prompts included in ALL-in-one PCA overviews to keep plots readable")

    parser.add_argument("--max_items_attn", type=int, default=25)
    parser.add_argument("--attn_keys", type=str, default=None,
                        help="Comma-separated prompt keys to visualize (e.g., 'instruction_original,instruct_polite_request')")
    parser.add_argument("--attn_keys_regex", type=str, default=None,
                        help="Regex for prompt keys to visualize (e.g., '.*polite.*')")

    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()
    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("compare08")

    items = load_instruction_json(args.instructions_json)
    if args.max_samples is not None and len(items) > args.max_samples:
        items = items[:args.max_samples]

    base, ft, tokenizer = build_model_and_tokenizer(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, args.device
    )

    dod_df = None
    subspace_df = None

    if args.run_diff_of_diffs:
        dod_df = run_diff_of_diffs(items, base, ft, tokenizer, args.layer_index, args.device, outdir, args.prompt_span)

    if args.run_paraphrase_subspace:
        subspace_df = paraphrase_pca_and_jacobian(items, ft, tokenizer, args.layer_index, args.device,
                                                  outdir, args.prompt_span, topk=args.topk_pca,
                                                  eps=args.eps_jacobian, scatter_labels=args.scatter_labels,
                                                  max_prompts_for_overview=args.max_prompts_for_overview)

    if args.run_attention_inspection:
        keys = [k.strip() for k in args.attn_keys.split(",")] if args.attn_keys else None
        attention_heatmaps(items, ft, tokenizer, args.layer_index, args.device, outdir, args.prompt_span,
                           max_items=args.max_items_attn, attn_keys=keys, attn_keys_regex=args.attn_keys_regex)

    write_markdown_report(outdir, args.layer_index, dod_df, subspace_df)
    log.info("Wrote report: %s", outdir / f"report_layer{args.layer_index}.md")


if __name__ == "__main__":
    main()
