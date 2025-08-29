
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

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


# Utils

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


# Data

@dataclass
class Item:
    prompt_count: int
    instruction_original: str
    paraphrases: Dict[str, str]
    input_text: str

    def all_prompt_variants(self, include_original: bool = True,
                            keys: Optional[Iterable[str]] = None,
                            regex: Optional[str] = None) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        if include_original:
            pairs.append(("instruction_original", self.instruction_original))
        for k, v in sorted(self.paraphrases.items()):
            if not k.startswith("instruct_"):
                continue
            if keys is not None and k not in keys:
                continue
            if regex is not None and re.search(regex, k) is None:
                continue
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


# Tokenisation

@dataclass
class Encoded:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_mask: torch.Tensor
    split_mask_input: Optional[torch.Tensor] = None  # tokens belonging to the "Input:" section (if present)


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

    # Try to locate "Input:" segment token span for stats (Pt 3). We simply find the substring boundary in text
    # and heuristically mark its token coverage.
    split_mask_input = None
    try:
        if "Input:" in text:
            before, after = text.split("Input:", 1)
            # token boundaries: encode both and subtract lengths
            enc_before = tokenizer(before, return_tensors="pt", add_special_tokens=True)
            Lb = enc_before["input_ids"][0].shape[0]
            split_mask_input = torch.zeros(T, dtype=torch.bool)
            split_mask_input[Lb:] = True  # rough estimate; good enough for aggregate stats
    except Exception:
        pass

    return Encoded(input_ids=input_ids.to(device),
                   attention_mask=attn_mask.to(device),
                   prompt_mask=prompt_mask.to(device),
                   split_mask_input=split_mask_input.to(device) if split_mask_input is not None else None)


# Model access & hooks

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
    resid_pre: List[torch.Tensor]    # [B,T,D]
    resid_post: List[torch.Tensor]   # [B,T,D]
    attn_probs: List[torch.Tensor]   # [H,T,T] per example


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


# Jacobian probe

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


# Representations & distances

def _rep_from_caps(enc: Encoded, caps: CapturedActivations, at: str) -> torch.Tensor:
    if at == "pre":
        x = caps.resid_pre[0][0]  # [T,D]
    else:
        x = caps.resid_post[0][0]
    return masked_mean(x, enc.prompt_mask)

def prompt_representation_from_layer(model, layer_index: int, enc: Encoded, rep_at: str) -> torch.Tensor:
    runner = capture_layer_activations(model, layer_index, next(model.parameters()).device)
    try:
        caps = runner([enc])
        if rep_at == "pre" and not caps.resid_pre:
            raise RuntimeError("Failed to capture resid_pre activations.")
        if rep_at == "post" and not caps.resid_post:
            raise RuntimeError("Failed to capture resid_post activations.")
        return _rep_from_caps(enc, caps, rep_at)
    finally:
        runner.hooks.remove_all()


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1.0 - cosine(a.unsqueeze(0), b.unsqueeze(0)).item())


# Model builders

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
    hybrid_both = copy.deepcopy(base_model).to(device)
    ft_blk = BlockAccessor(ft_model, layer_index)
    ha_blk = BlockAccessor(hybrid_attn, layer_index)
    hm_blk = BlockAccessor(hybrid_mlp,  layer_index)
    hb_blk = BlockAccessor(hybrid_both, layer_index)
    ha_blk.swap_attention_from(ft_blk)
    hm_blk.swap_mlp_from(ft_blk)
    # both:
    hb_blk.swap_attention_from(ft_blk)
    hb_blk.swap_mlp_from(ft_blk)
    for m in [hybrid_attn, hybrid_mlp, hybrid_both]:
        m.eval()
    return hybrid_attn, hybrid_mlp, hybrid_both


# 1) Diff-of-diffs (component swaps; rep-level)

def diff_of_diffs_for_item(
    models: Dict[str, nn.Module],
    tokenizer,
    item: Item,
    layer_index: int,
    device: str,
    prompt_span: str,
    rep_at: str,
) -> Dict[str, float]:
    orig_text = build_prompt_text(item.instruction_original, item.input_text)
    enc_orig = encode_prompt(tokenizer, orig_text, device, prompt_span=prompt_span)

    pps = [(k, v) for k, v in item.paraphrases.items() if k.startswith("instruct_")]
    if len(pps) == 0:
        return {}

    reps_orig = {name: prompt_representation_from_layer(model, layer_index, enc_orig, rep_at)
                 for name, model in models.items()}

    dists = {name: [] for name in models.keys()}
    for key, para in pps:
        enc = encode_prompt(tokenizer, build_prompt_text(para, item.input_text), device, prompt_span=prompt_span)
        for name, model in models.items():
            rep = prompt_representation_from_layer(model, layer_index, enc, rep_at)
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
) -> Dict[str, pd.DataFrame]:
    hyb_attn, hyb_mlp, hyb_both = build_hybrid_models(base, ft, layer_index, device)
    models = {"BASE": base, "FT": ft, "HYB_ATTN": hyb_attn, "HYB_MLP": hyb_mlp, "HYB_BOTH": hyb_both}

    out = {}
    for rep_at in ["pre", "post"]:
        rows = []
        for item in items:
            row = {"prompt_count": item.prompt_count}
            row.update(diff_of_diffs_for_item(models, tokenizer, item, layer_index, device, prompt_span, rep_at))
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("prompt_count")
        df.to_csv(outdir / f"diff_of_diffs_layer{layer_index}_{rep_at}.csv", index=False)

        # Summary plot
        fig, ax = plt.subplots(figsize=(7.5, 4))
        labels = ["FT", "HYB_ATTN", "HYB_MLP", "HYB_BOTH"]
        means = []
        Ns = []
        for col in labels:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            Ns.append(int(s.shape[0]))
            means.append(s.mean() if not s.empty else 0.0)
        ax.bar(labels, means)
        ax.set_title(f"Diff-of-diffs (avg across items) — layer {layer_index} ({rep_at})\nN={Ns}")
        ax.set_ylabel("Δ = D_base - D_model (cosine distance)")
        fig.tight_layout()
        fig.savefig(outdir / f"diff_of_diffs_layer{layer_index}_{rep_at}_summary.png", dpi=180)
        plt.close(fig)
        out[rep_at] = df
    return out


# 2) Paraphrase subspace shrinkage (multi-model PCA + Jacobian)

def paraphrase_pca_and_jacobian_for_model(
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
    model_tag: str = "FT",
    max_prompts_for_overview: int = 200,
) -> pd.DataFrame:
    """
    Aggregated plotting:
      - ONE overlay scree plot with all prompts (different colors).
      - ONE average scree with 95% CI.
      - ONE overlay PCA scatter (PC1 vs PC2 per prompt frame) with all prompts, colored by prompt.
      - ONE Jacobian overlay plot (each prompt line across PC/RANDOM/MEAN).
      - ONE Jacobian averages bar with 95% CI.
    No per-prompt figure files are written.
    """
    f_mlp = mlp_function_from_block(model, layer_index)

    summary_rows = []

    # Collectors for aggregate plots
    scree_list = []     # list of np.array varexpl per prompt
    scree_ids  = []     # matching prompt_count
    scatter_pts = []    # list of (pc1, pc2, prompt_count)
    jac_points = []     # list of (prompt_count, jac_PC_mean, jac_RANDOM_mean, jac_MEAN)

    for item in items:
        H_list = []
        keys = []
        for key, para in item.all_prompt_variants(include_original=True):
            if not key.startswith("instruct_") and key != "instruction_original":
                continue
            enc = encode_prompt(tokenizer, build_prompt_text(para, item.input_text), device, prompt_span=prompt_span)
            runner = capture_layer_activations(model, layer_index, next(model.parameters()).device)
            try:
                caps = runner([enc])
                if not caps.resid_pre:
                    continue
                pre = caps.resid_pre[0][0]
                h = masked_mean(pre, enc.prompt_mask)
            finally:
                runner.hooks.remove_all()
            H_list.append(h.cpu().numpy().astype(np.float32))
            keys.append(key)
        if len(H_list) < 3:
            continue
        H = np.stack(H_list, axis=0)  # [N, D]

        # PCA
        H_mean = H.mean(axis=0, keepdims=True)
        Hc = H - H_mean
        U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
        var = (S ** 2)
        if var.sum() <= 0:
            continue
        varexpl = var / var.sum()
        scree_list.append(varexpl)
        scree_ids.append(item.prompt_count)

        # 2D scatter in per-prompt frame
        PCs = Vt[:2]              # [2, D]
        proj = Hc @ PCs.T         # [N, 2]
        for i in range(proj.shape[0]):
            scatter_pts.append((float(proj[i,0]), float(proj[i,1]), item.prompt_count))

        # Directional Jacobian (average across H)
        k = int(min(topk, Vt.shape[0]))
        PCs_k = Vt[:k]  # [k,D]
        model_dtype = next(model.parameters()).dtype
        PCs_t = torch.from_numpy(PCs_k).to(device).to(model_dtype)
        h_tensor = torch.from_numpy(H).to(device).to(model_dtype)  # [N,D]
        H_mean_t = torch.from_numpy(H_mean.squeeze(0)).to(device).to(model_dtype)
        PCs_t = F.normalize(PCs_t, dim=-1)
        rand_dirs = torch.randn_like(PCs_t); rand_dirs = F.normalize(rand_dirs, dim=-1)
        mean_dir = F.normalize(H_mean_t, dim=-1, eps=1e-8).unsqueeze(0)

        norms_pcs_list, norms_rand_list, norms_mean_list = [], [], []
        for i in range(h_tensor.shape[0]):
            hi = h_tensor[i]
            mlp_f = mlp_function_from_block(model, layer_index)
            norms_pcs_list.append(jacobian_directional_norms(mlp_f, hi, PCs_t, eps=eps).cpu().numpy())
            norms_rand_list.append(jacobian_directional_norms(mlp_f, hi, rand_dirs, eps=eps).cpu().numpy())
            norms_mean_list.append(jacobian_directional_norms(mlp_f, hi, mean_dir, eps=eps).cpu().numpy())
        norms_pcs = np.stack(norms_pcs_list, axis=0).mean(axis=0)   # [k]
        norms_rand = np.stack(norms_rand_list, axis=0).mean(axis=0) # [k]
        norms_mean = np.stack(norms_mean_list, axis=0).mean(axis=0) # [1]

        jac_PC_mean = float(norms_pcs.mean())
        jac_RANDOM_mean = float(norms_rand.mean())
        jac_MEAN = float(norms_mean.mean())
        jac_points.append((item.prompt_count, jac_PC_mean, jac_RANDOM_mean, jac_MEAN))

        # Summary row (kept for CSV aggregation across models)
        summary_rows.append({
            "model": model_tag,
            "prompt_count": item.prompt_count,
            "N_paraphrases": int(H.shape[0]),
            "k_top": int(k),
            "pc1_var_expl": float(varexpl[0]) if len(varexpl) else np.nan,
            "cum_var_expl_topk": float(varexpl[:k].sum()) if len(varexpl) else np.nan,
            "jac_PC_mean": jac_PC_mean,
            "jac_RANDOM_mean": jac_RANDOM_mean,
            "jac_MEAN": jac_MEAN,
            "contraction_ratio_PC_over_RANDOM": float(jac_PC_mean / jac_RANDOM_mean) if jac_RANDOM_mean != 0 else np.nan,
        })

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows).sort_values(["model","prompt_count"])

    # Aggregate plots (ONE overlay + ONE average w/ CI)
    # Scree overlay (truncate to a common max component length)
    if len(scree_list) > 0:
        max_len = min(max(len(v) for v in scree_list), 64)  # cap to keep readable
        xs = np.arange(1, max_len+1)

        # Overlay
        plt.figure(figsize=(8,5))
        for ve, pid in zip(scree_list, scree_ids):
            y = ve[:max_len]
            plt.plot(xs, y, alpha=0.6, label=f"p{pid}")
        plt.xlabel("component"); plt.ylabel("variance explained")
        plt.title(f"[{model_tag}] PCA scree — ALL prompts (layer {layer_index})")
        if len(scree_list) <= 20: plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / f"{model_tag}_pca_scree_ALL_overlay_layer{layer_index}.png", dpi=180)
        plt.close()

        # Average with 95% CI
        # pad by NaNs to common length, then nanmean
        M = len(scree_list)
        mat = np.full((M, max_len), np.nan, dtype=np.float32)
        for i, ve in enumerate(scree_list):
            mat[i, :min(max_len, len(ve))] = ve[:max_len]
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat, axis=0, ddof=1)
        n_eff = np.sum(~np.isnan(mat), axis=0).clip(min=1)
        ci95 = 1.96 * std / np.sqrt(n_eff)

        plt.figure(figsize=(8,5))
        plt.plot(xs, mean, lw=2)
        plt.fill_between(xs, mean-ci95, mean+ci95, alpha=0.25)
        plt.xlabel("component"); plt.ylabel("variance explained")
        plt.title(f"[{model_tag}] PCA scree — mean ±95% CI (layer {layer_index})")
        plt.tight_layout()
        plt.savefig(outdir / f"{model_tag}_pca_scree_ALL_meanCI_layer{layer_index}.png", dpi=180)
        plt.close()

    # PCA scatter overlay + radius mean plot
    if len(scatter_pts) > 0:
        df_sc = pd.DataFrame(scatter_pts, columns=["pc1","pc2","prompt_count"])
        # Overlay scatter colored by prompt
        plt.figure(figsize=(8,6))
        for pid, sub in df_sc.groupby("prompt_count"):
            plt.scatter(sub["pc1"], sub["pc2"], s=12, alpha=0.6, label=f"p{pid}")
        plt.xlabel("PC1 (per-prompt frame)"); plt.ylabel("PC2 (per-prompt frame)")
        plt.title(f"[{model_tag}] Paraphrase PCA scatter — ALL prompts (layer {layer_index})")
        if df_sc["prompt_count"].nunique() <= 20:
            plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / f"{model_tag}_pca_scatter_ALL_overlay_layer{layer_index}.png", dpi=200)
        plt.close()

        # Mean radius per prompt, then global mean±CI
        df_sc["r2"] = df_sc["pc1"]**2 + df_sc["pc2"]**2
        by_prompt = df_sc.groupby("prompt_count")["r2"].mean()
        mu = by_prompt.mean()
        std = by_prompt.std(ddof=1)
        n = max(len(by_prompt), 1)
        ci95 = 1.96 * (std / np.sqrt(n)) if n > 1 else 0.0

        plt.figure(figsize=(6.5,4.2))
        plt.bar([0], [mu], yerr=[ci95], width=0.6)
        plt.xticks([0], ["mean radius^2"])
        plt.ylabel("mean r^2 across prompts")
        plt.title(f"[{model_tag}] PCA scatter size — mean ±95% CI (layer {layer_index})")
        plt.tight_layout()
        plt.savefig(outdir / f"{model_tag}_pca_scatter_radius_meanCI_layer{layer_index}.png", dpi=180)
        plt.close()

    # Jacobian overlay + averages
    if len(jac_points) > 0:
        dfj = pd.DataFrame(jac_points, columns=["prompt_count","jac_PC_mean","jac_RANDOM_mean","jac_MEAN"])

        # Overlay: each prompt is a line across categories
        cats = ["PC","RANDOM","MEAN"]
        plt.figure(figsize=(7.5,5))
        for _, r in dfj.iterrows():
            ys = [r["jac_PC_mean"], r["jac_RANDOM_mean"], r["jac_MEAN"]]
            plt.plot(cats, ys, alpha=0.5)
        plt.ylabel("‖(f(h+εd)-f(h-εd))‖ / (2ε)")
        plt.title(f"[{model_tag}] MLP directional Jacobian — ALL prompts overlay (layer {layer_index})")
        plt.tight_layout()
        plt.savefig(outdir / f"{model_tag}_jacobian_ALL_overlay_layer{layer_index}.png", dpi=180)
        plt.close()

        # Averages with 95% CI
        means = dfj[["jac_PC_mean","jac_RANDOM_mean","jac_MEAN"]].mean(axis=0)
        stds  = dfj[["jac_PC_mean","jac_RANDOM_mean","jac_MEAN"]].std(axis=0, ddof=1)
        n = len(dfj)
        ci95 = 1.96 * (stds / np.sqrt(n)) if n > 1 else 0.0
        x = np.arange(3)
        plt.figure(figsize=(7.5,5))
        plt.bar(x, means.values, yerr=ci95 if np.isscalar(ci95) else ci95.values, width=0.6)
        plt.xticks(x, ["PC","RANDOM","MEAN"])
        plt.ylabel("‖(f(h+εd)-f(h-εd))‖ / (2ε)")
        plt.title(f"[{model_tag}] MLP directional Jacobian — mean ±95% CI (layer {layer_index})")
        plt.tight_layout()
        plt.savefig(outdir / f"{model_tag}_jacobian_ALL_meanCI_layer{layer_index}.png", dpi=180)
        plt.close()

    return summary_df


def paraphrase_pca_and_jacobian_multi(
    items: List[Item],
    models: Dict[str, nn.Module],
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
    dfs = []
    for tag, model in models.items():
        df = paraphrase_pca_and_jacobian_for_model(
            items, model, tokenizer, layer_index, device, outdir, prompt_span,
            topk=topk, eps=eps, scatter_labels=scatter_labels, model_tag=tag,
            max_prompts_for_overview=max_prompts_for_overview
        )
        if not df.empty:
            dfs.append(df)
            df.to_csv(outdir / f"{tag}_paraphrase_subspace_jacobian_layer{layer_index}.csv", index=False)
    if not dfs:
        return pd.DataFrame()

    cat = pd.concat(dfs, axis=0, ignore_index=True)
    cat.to_csv(outdir / f"ALL_paraphrase_subspace_jacobian_layer{layer_index}.csv", index=False)

    # Global comparison plots across models (means only)
    def _bar(metric: str, fname: str, ylim: Optional[Tuple[float,float]] = None):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        order = list(models.keys())
        means = [pd.to_numeric(cat.loc[cat["model"]==m, metric], errors="coerce").mean() for m in order]
        stds  = [pd.to_numeric(cat.loc[cat["model"]==m, metric], errors="coerce").std(ddof=1) for m in order]
        ns    = [cat.loc[cat["model"]==m, metric].notna().sum() for m in order]
        ci95  = [ (1.96 * stds[i] / np.sqrt(ns[i])) if ns[i] > 1 else 0.0 for i in range(len(order)) ]
        ax.bar(order, means, yerr=ci95, width=0.7)
        ax.set_title(f"{metric} — averaged across prompts (layer {layer_index})")
        ax.set_ylabel(metric.replace("_"," "))
        if ylim is not None:
            ax.set_ylim(*ylim)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=180)
        plt.close(fig)

    _bar("contraction_ratio_PC_over_RANDOM", f"subspace_CONTRACTION_ratio_layer{layer_index}.png", ylim=(0,1.5))
    _bar("jac_PC_mean", f"subspace_jac_PC_mean_layer{layer_index}.png")
    _bar("jac_RANDOM_mean", f"subspace_jac_RANDOM_mean_layer{layer_index}.png")
    _bar("jac_MEAN", f"subspace_jac_MEAN_layer{layer_index}.png")
    _bar("pc1_var_expl", f"subspace_pc1_var_expl_layer{layer_index}.png")
    _bar("cum_var_expl_topk", f"subspace_cum_var_topk_layer{layer_index}.png")

    return cat


# 3) Attention inspection (grids + numeric token stats)

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
    # NOTE: to avoid hundreds of per-prompt images, we skip per-example grids here
    # Users can enable token stats or similarity for aggregate views
    return


def _flatten_attn(A: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is not None and mask.dtype == torch.bool and mask.shape[0] == A.shape[0]:
        idx = mask.nonzero(as_tuple=True)[0]
        idx = idx.to(A.device)
        A = A[idx][:, idx]
    return A.reshape(-1)

def attention_similarity(
    items: List[Item],
    base: nn.Module,
    ft: nn.Module,
    tokenizer,
    layer_index: int,
    device: str,
    outdir: Path,
    prompt_span: str,
    max_items: int = 100,
    attn_keys: Optional[List[str]] = None,
    attn_keys_regex: Optional[str] = None,
    metric_kl: bool = True,
    eps: float = 1e-8,
) -> pd.DataFrame:
    rgx = re.compile(attn_keys_regex) if attn_keys_regex else None

    rows = []
    count = 0
    for item in items:
        for key, para in item.all_prompt_variants(include_original=True):
            if attn_keys is not None and len(attn_keys) and key not in attn_keys: continue
            if rgx is not None and not rgx.search(key): continue
            if count >= max_items: break

            text = build_prompt_text(para, item.input_text)
            enc = encode_prompt(tokenizer, text, device, prompt_span=prompt_span)

            # BASE
            run_b = capture_layer_activations(base, layer_index, device)
            caps_b = run_b([enc]); run_b.hooks.remove_all()
            if not caps_b.attn_probs: continue
            Ab = caps_b.attn_probs[0]  # [H,T,T]

            # FT
            run_f = capture_layer_activations(ft, layer_index, device)
            caps_f = run_f([enc]); run_f.hooks.remove_all()
            if not caps_f.attn_probs: continue
            Af = caps_f.attn_probs[0]  # [H,T,T]

            H, T, _ = Ab.shape
            cos_list, skl_list = [], []

            for h in range(H):
                vb = _flatten_attn(Ab[h], mask=enc.prompt_mask)
                vf = _flatten_attn(Af[h], mask=enc.prompt_mask)
                cb = F.normalize(vb.float(), dim=0)
                cf = F.normalize(vf.float(), dim=0)
                cos_sim = torch.clamp((cb * cf).sum(), -1.0, 1.0).item()
                cos_list.append(cos_sim)

                if metric_kl:
                    pb = (vb.float() + eps) / (vb.float().sum() + eps * vb.numel())
                    pf = (vf.float() + eps) / (vf.float().sum() + eps * vf.numel())
                    kl_bf = (pb * (pb / pf).log()).sum()
                    kl_fb = (pf * (pf / pb).log()).sum()
                    skl = float((kl_bf + kl_fb).item())
                    skl_list.append(skl)

            row = {"prompt_count": item.prompt_count, "key": key}
            for h in range(H):
                row[f"cos_head{h}"] = cos_list[h]
                if metric_kl:
                    row[f"skl_head{h}"] = skl_list[h]
            row["cos_mean"] = float(np.mean(cos_list))
            if metric_kl:
                row["skl_mean"] = float(np.mean(skl_list))
            rows.append(row)
            count += 1
        if count >= max_items: break

    df = pd.DataFrame(rows)
    csv_path = outdir / f"attn_similarity_layer{layer_index}.csv"
    if not df.empty:
        df.to_csv(csv_path, index=False)

        # Per-head cosine summary with 95% CI
        head_cols = [c for c in df.columns if re.match(r"^cos_head\d+$", c)]
        head_ids = [int(c.split("head")[1]) for c in head_cols]
        head_means = [df[c].mean() for c in head_cols]
        head_stds  = [df[c].std(ddof=1) for c in head_cols]
        ns         = [df[c].notna().sum() for c in head_cols]
        ci95       = [ (1.96 * head_stds[i] / np.sqrt(ns[i])) if ns[i] > 1 else 0.0 for i in range(len(head_cols)) ]

        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar([f"h{h}" for h in head_ids], head_means, yerr=ci95)
        ax.set_ylim(0,1)
        ax.set_title(f"Attention cosine similarity (BASE vs FT) per head — layer {layer_index}")
        ax.set_ylabel("cosine similarity")
        fig.tight_layout()
        fig.savefig(outdir / f"attn_similarity_layer{layer_index}_cosine_per_head.png", dpi=180)
        plt.close(fig)

        # Heatmap remains a single combined figure (not per-prompt)
        fig, ax = plt.subplots(figsize=(max(6, len(head_cols)*0.5), max(6, len(df)*0.25)))
        im = ax.imshow(df[head_cols].values, aspect="auto", interpolation="nearest", origin="lower")
        ax.set_yticks(np.arange(len(df))); ax.set_yticklabels([f"{r.prompt_count}:{r.key}" for _, r in df.iterrows()], fontsize=7)
        ax.set_xticks(np.arange(len(head_cols))); ax.set_xticklabels(head_cols, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Attention cosine similarity (BASE vs FT) — layer {layer_index}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(outdir / f"attn_similarity_layer{layer_index}_cosine_heatmap.png", dpi=200)
        plt.close(fig)

        skl_cols = [c for c in df.columns if re.match(r"^skl_head\d+$", c)]
        if skl_cols:
            skl_means = [df[c].mean() for c in skl_cols]
            skl_stds  = [df[c].std(ddof=1) for c in skl_cols]
            ns2       = [df[c].notna().sum() for c in skl_cols]
            ci95_skl  = [ (1.96 * skl_stds[i] / np.sqrt(ns2[i])) if ns2[i] > 1 else 0.0 for i in range(len(skl_cols)) ]

            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar([f"h{h}" for h in head_ids], skl_means, yerr=ci95_skl)
            ax.set_title(f"Attention symmetric KL (BASE vs FT) per head — layer {layer_index}")
            ax.set_ylabel("symmetric KL")
            fig.tight_layout()
            fig.savefig(outdir / f"attn_similarity_layer{layer_index}_skl_per_head.png", dpi=180)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(max(6, len(skl_cols)*0.5), max(6, len(df)*0.25)))
            im = ax.imshow(df[skl_cols].values, aspect="auto", interpolation="nearest", origin="lower")
            ax.set_yticks(np.arange(len(df))); ax.set_yticklabels([f"{r.prompt_count}:{r.key}" for _, r in df.iterrows()], fontsize=7)
            ax.set_xticks(np.arange(len(skl_cols))); ax.set_xticklabels(skl_cols, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"Attention symmetric KL (BASE vs FT) — layer {layer_index}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(outdir / f"attn_similarity_layer{layer_index}_skl_heatmap.png", dpi=200)
            plt.close(fig)

    return df


def attention_token_stats(
    items: List[Item],
    model: nn.Module,
    tokenizer,
    layer_index: int,
    device: str,
    outdir: Path,
    prompt_span: str,
    max_items: int = 200,
    attn_keys_regex: Optional[str] = None,
) -> pd.DataFrame:
    rgx = re.compile(attn_keys_regex) if attn_keys_regex else None
    rows = []
    count = 0
    for item in items:
        for key, para in item.all_prompt_variants(include_original=True):
            if rgx is not None and not rgx.search(key):
                continue
            if count >= max_items:
                break
            enc = encode_prompt(tokenizer, build_prompt_text(para, item.input_text), device, prompt_span=prompt_span)
            runner = capture_layer_activations(model, layer_index, device)
            caps = runner([enc]); runner.hooks.remove_all()
            if not caps.attn_probs:
                continue
            A = caps.attn_probs[0]  # [H,T,T]
            H, T, _ = A.shape
            avg = A.mean(0)         # [T,T]
            with torch.no_grad():
                m_first = avg[:,0].mean().item()
                m_last  = avg[:,-1].mean().item()
                diag    = torch.diag(avg).mean().item()
                lower   = torch.tril(avg, diagonal=-1).sum().item() / (T*(T-1)/2)
                upper   = torch.triu(avg, diagonal=+1).sum().item() / (T*(T-1)/2)
                input_mass = np.nan
                if enc.split_mask_input is not None:
                    idx = enc.split_mask_input.nonzero(as_tuple=True)[0]
                    if idx.numel() > 0:
                        input_mass = avg[:, idx].mean().item()
            rows.append({
                "prompt_count": item.prompt_count, "key": key,
                "mass_first": m_first, "mass_last": m_last,
                "mass_diag": diag, "mass_lower": lower, "mass_upper": upper,
                "mass_into_input": input_mass,
                "T": T
            })
            count += 1
        if count >= max_items:
            break
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(outdir / f"attention_token_stats_layer{layer_index}.csv", index=False)
        # Also provide a summary plot with 95% CI for each metric
        metrics = ["mass_first","mass_last","mass_diag","mass_lower","mass_upper"]
        means = [df[m].mean() for m in metrics]
        stds  = [df[m].std(ddof=1) for m in metrics]
        ns    = [df[m].notna().sum() for m in metrics]
        ci95  = [ (1.96 * stds[i] / np.sqrt(ns[i])) if ns[i] > 1 else 0.0 for i in range(len(metrics)) ]
        x = np.arange(len(metrics))
        plt.figure(figsize=(8,4.5))
        plt.bar(x, means, yerr=ci95, width=0.7)
        plt.xticks(x, metrics, rotation=20)
        plt.title(f"Attention token stats — mean ±95% CI (layer {layer_index})")
        plt.tight_layout()
        plt.savefig(outdir / f"attention_token_stats_layer{layer_index}_meanCI.png", dpi=180)
        plt.close()
    return df


# 4) Actual run with weight patching (generation-level Δ)

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
    if not text:
        text = "."
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True, return_dict=True)
        last = outputs.hidden_states[-1][0]  # [T, D]
    return last.float().mean(dim=0)          # [D]

def run_weight_patching_generation_eval(
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
    hyb_attn, hyb_mlp, hyb_both = build_hybrid_models(base, ft, layer_index, device)
    MODELS = {"BASE": base, "FT": ft, "HYB_ATTN": hyb_attn, "HYB_MLP": hyb_mlp, "HYB_BOTH": hyb_both}
    decode_cfg = f"max_new_tokens={max_new_tokens}, do_sample={do_sample}, temperature={temperature}, top_p={top_p}, top_k={top_k}"
    (outdir / "decode_config.txt").write_text(decode_cfg, encoding="utf-8")

    rows_gen = []
    rows_metrics = []

    processed = 0
    for item in items:
        if processed >= max_items:
            break
        variants = item.all_prompt_variants(include_original=True, keys=keys, regex=keys_regex)
        if len(variants) < 2:
            continue

        # original
        orig_answers: Dict[str, str] = {}
        orig_embeds: Dict[str, torch.Tensor] = {}
        orig_prompt = build_prompt_text(item.instruction_original, item.input_text)
        for mname, model in MODELS.items():
            ans = generate_answer(model, tokenizer, orig_prompt, device,
                                  max_new_tokens, do_sample, temperature, top_p, top_k)
            emb = text_embedding(model, tokenizer, ans, device)
            orig_answers[mname] = ans
            orig_embeds[mname]  = emb
            rows_gen.append({"prompt_count": item.prompt_count,"key": "instruction_original","model": mname,"answer": ans})

        # paraphrases
        for key, phr in variants:
            if key == "instruction_original":
                continue
            ptxt = build_prompt_text(phr, item.input_text)
            for mname, model in MODELS.items():
                ans = generate_answer(model, tokenizer, ptxt, device,
                                      max_new_tokens, do_sample, temperature, top_p, top_k)
                rows_gen.append({"prompt_count": item.prompt_count,"key": key,"model": mname,"answer": ans})

        # distances
        df_this = pd.DataFrame([r for r in rows_gen if r["prompt_count"] == item.prompt_count])
        for mname in MODELS.keys():
            emb_orig = orig_embeds[mname]
            dlist = []
            for _, r in df_this[(df_this["model"] == mname) & (df_this["key"] != "instruction_original")].iterrows():
                emb_para = text_embedding(MODELS[mname], tokenizer, r["answer"], device)
                dlist.append(cosine_distance(emb_orig, emb_para))
            D_model = float(np.mean(dlist)) if dlist else np.nan
            rows_metrics.append({"prompt_count": item.prompt_count,"model": mname,"D_model_answer": D_model,"N_paraphrases": len(dlist)})
        processed += 1

    df_gen = pd.DataFrame(rows_gen); df_gen.to_csv(outdir / "generations.csv", index=False)
    df_metrics = pd.DataFrame(rows_metrics); df_metrics.to_csv(outdir / "answer_distance_per_model.csv", index=False)

    pivot = df_metrics.pivot(index="prompt_count", columns="model", values="D_model_answer").reset_index()
    for col in ["BASE","FT","HYB_ATTN","HYB_MLP","HYB_BOTH"]:
        if col not in pivot.columns: pivot[col] = np.nan
    pivot["DELTA_FT"]       = pivot["BASE"] - pivot["FT"]
    pivot["DELTA_HYB_ATTN"] = pivot["BASE"] - pivot["HYB_ATTN"]
    pivot["DELTA_HYB_MLP"]  = pivot["BASE"] - pivot["HYB_MLP"]
    pivot["DELTA_HYB_BOTH"] = pivot["BASE"] - pivot["HYB_BOTH"]
    pivot.to_csv(outdir / "answer_diff_of_diffs.csv", index=False)

    # Mean Δ with 95% CI
    labels = ["FT","HYB_ATTN","HYB_MLP","HYB_BOTH"]
    means  = [pd.to_numeric(pivot[f"DELTA_{k}"], errors="coerce").mean() for k in labels]
    stds   = [pd.to_numeric(pivot[f"DELTA_{k}"], errors="coerce").std(ddof=1) for k in labels]
    ns     = [pd.to_numeric(pivot[f"DELTA_{k}"], errors="coerce").notna().sum() for k in labels]
    ci95   = [ (1.96 * stds[i] / np.sqrt(ns[i])) if ns[i] > 1 else 0.0 for i in range(len(labels)) ]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(labels, means, yerr=ci95)
    ax.set_title(f"Answer-level robustness Δ vs BASE (layer {layer_index})")
    ax.set_ylabel("Δ = D_base - D_model (higher is better)")
    fig.tight_layout()
    fig.savefig(outdir / "answer_diff_of_diffs_summary.png", dpi=180)
    plt.close(fig)

    # Per-prompt grouped bars (single figure)
    fig, ax = plt.subplots(figsize=(10.5, 5))
    x = np.arange(pivot.shape[0])
    width = 0.22
    ax.bar(x - 1.5*width, pd.to_numeric(pivot["DELTA_FT"], errors="coerce"), width, label="FT")
    ax.bar(x - 0.5*width, pd.to_numeric(pivot["DELTA_HYB_ATTN"], errors="coerce"), width, label="HYB_ATTN")
    ax.bar(x + 0.5*width, pd.to_numeric(pivot["DELTA_HYB_MLP"], errors="coerce"), width, label="HYB_MLP")
    ax.bar(x + 1.5*width, pd.to_numeric(pivot["DELTA_HYB_BOTH"], errors="coerce"), width, label="HYB_BOTH")
    ax.set_xticks(x); ax.set_xticklabels([str(int(i)) for i in pivot["prompt_count"].fillna(-1)])
    ax.set_xlabel("prompt_count"); ax.set_ylabel("Δ (higher is better)")
    ax.set_title(f"Answer-level Δ per prompt (layer {layer_index})")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "answer_diff_of_diffs_per_prompt.png", dpi=200)
    plt.close(fig)

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
    (outdir / "weight_patching_report.md").write_text("\n".join(report_lines), encoding="utf-8")


# Markdown report

def write_markdown_report(
    outdir: Path,
    layer_index: int,
    dod_pre: Optional[pd.DataFrame],
    dod_post: Optional[pd.DataFrame],
    subspace_df: Optional[pd.DataFrame],
    attn_sim_df: Optional[pd.DataFrame],
    token_stats_df: Optional[pd.DataFrame],
    did_gen_eval: bool,
    models_for_subspace: List[str],
):
    md = []
    def add(s=""): md.append(s)

    add(f"# Paraphrase-robustness analysis — layer {layer_index}")
    add("This report lists the generated files and summary metrics.")
    add("")

    add("## 1) Causal component replacement (diff-of-diffs on prompt representations)")
    for rep_at in ["pre", "post"]:
        csv = outdir / f"diff_of_diffs_layer{layer_index}_{rep_at}.csv"
        png = outdir / f"diff_of_diffs_layer{layer_index}_{rep_at}_summary.png"
        add(f"- Representation position: **{rep_at}**")
        if csv.exists(): add(f"  - CSV: `{csv.name}`")
        if png.exists(): add(f"  - Summary: `{png.name}`")
    add("*(If pre≈0 while post>0, it indicates the change lives **inside** the measured layer.)*")
    add("")

    add("## 2) Paraphrase-subspace shrinkage (PCA + Jacobian; multi-model)")
    acsv = outdir / f"ALL_paraphrase_subspace_jacobian_layer{layer_index}.csv"
    if acsv.exists():
        add(f"- Combined CSV: `{acsv.name}`")
        for tag in models_for_subspace:
            per = outdir / f"{tag}_paraphrase_subspace_jacobian_layer{layer_index}.csv"
            if per.exists():
                add(f"  - Per-model: `{per.name}`")
        for f in [
            f"subspace_CONTRACTION_ratio_layer{layer_index}.png",
            f"subspace_jac_PC_mean_layer{layer_index}.png",
            f"subspace_jac_RANDOM_mean_layer{layer_index}.png",
            f"subspace_jac_MEAN_layer{layer_index}.png",
            f"subspace_pc1_var_expl_layer{layer_index}.png",
            f"subspace_cum_var_topk_layer{layer_index}.png",
        ]:
            if (outdir / f).exists():
                add(f"  - `{f}`")
        # Per-model aggregate figures (overlay/meanCI) are saved with [{MODEL}]_... filenames.
    add("")

    add("## 3) Attention patterns")
    sim_csv = outdir / f"attn_similarity_layer{layer_index}.csv"
    if sim_csv.exists():
        add(f"- Similarity CSV: `{sim_csv.name}`")
        for f in [
            f"attn_similarity_layer{layer_index}_cosine_per_head.png",
            f"attn_similarity_layer{layer_index}_cosine_heatmap.png",
            f"attn_similarity_layer{layer_index}_skl_per_head.png",
            f"attn_similarity_layer{layer_index}_skl_heatmap.png",
        ]:
            if (outdir / f).exists():
                add(f"  - `{f}`")
    tok_csv = outdir / f"attention_token_stats_layer{layer_index}.csv"
    if tok_csv.exists():
        add(f"- Token-pattern stats CSV: `{tok_csv.name}`")
        mfig = outdir / f"attention_token_stats_layer{layer_index}_meanCI.png"
        if mfig.exists(): add(f"  - `{mfig.name}`")
    add("")

    add("## 4) Answer-level weight patching (actual generations)")
    if did_gen_eval:
        for f in [
            "generations.csv",
            "answer_distance_per_model.csv",
            "answer_diff_of_diffs.csv",
            "answer_diff_of_diffs_summary.png",
            "answer_diff_of_diffs_per_prompt.png",
            "weight_patching_report.md",
        ]:
            if (outdir / f).exists():
                add(f"- `{f}`")
    else:
        add("- (Skipped; run with `--run_weight_patching_eval` to produce.)")

    (outdir / f"report_layer{layer_index}.md").write_text("\n".join(md), encoding="utf-8")


# CLI

def main():
    parser = argparse.ArgumentParser(description="Paraphrase robustness analyses (expert-augmented, aggregated outputs)")

    parser.add_argument("--instructions_json", type=str, required=True, help="Path to instructions JSON")

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
    parser.add_argument("--run_attention_similarity", action="store_true")
    parser.add_argument("--run_attention_token_stats", action="store_true")
    parser.add_argument("--run_weight_patching_eval", action="store_true")

    parser.add_argument("--topk_pca", type=int, default=8)
    parser.add_argument("--eps_jacobian", type=float, default=1e-3)
    parser.add_argument("--scatter_labels", action="store_true")
    parser.add_argument("--max_prompts_for_overview", type=int, default=200)
    parser.add_argument("--subspace_models", type=str, default="BASE,FT,HYB_MLP")

    parser.add_argument("--max_items_attn", type=int, default=25)
    parser.add_argument("--attn_keys", type=str, default=None)
    parser.add_argument("--attn_keys_regex", type=str, default=None)

    parser.add_argument("--max_items_gen", type=int, default=50)
    parser.add_argument("--decode_max_new_tokens", type=int, default=256)
    parser.add_argument("--decode_do_sample", action="store_true")
    parser.add_argument("--decode_temperature", type=float, default=0.0)
    parser.add_argument("--decode_top_p", type=float, default=1.0)
    parser.add_argument("--decode_top_k", type=int, default=0)

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--keys", type=str, default=None)
    parser.add_argument("--keys_regex", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("compare09")

    items = load_instruction_json(args.instructions_json)
    if args.max_samples is not None and len(items) > args.max_samples:
        items = items[:args.max_samples]

    base, ft, tokenizer = build_model_and_tokenizer(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, args.device
    )

    hyb_attn, hyb_mlp, hyb_both = build_hybrid_models(base, ft, args.layer_index, args.device)
    MODELS = {"BASE": base, "FT": ft, "HYB_ATTN": hyb_attn, "HYB_MLP": hyb_mlp, "HYB_BOTH": hyb_both}

    keys = [k.strip() for k in args.keys.split(",")] if args.keys else None

    # 1) Diff-of-diffs (rep level)
    dod = {"pre": None, "post": None}
    if args.run_diff_of_diffs:
        dod = run_diff_of_diffs(items, base, ft, tokenizer, args.layer_index, args.device, outdir, args.prompt_span)
        if dod["pre"] is not None:  log.info("Saved diff-of-diffs CSV/plots (pre).")
        if dod["post"] is not None: log.info("Saved diff-of-diffs CSV/plots (post).")

    # 2) Subspace shrinkage (multi-model, aggregated plots)
    subspace_df = None
    if args.run_paraphrase_subspace:
        tags = [t.strip() for t in args.subspace_models.split(",") if t.strip() in MODELS]
        mm = {t: MODELS[t] for t in tags}
        subspace_df = paraphrase_pca_and_jacobian_multi(
            items, mm, tokenizer, args.layer_index, args.device, outdir, args.prompt_span,
            topk=args.topk_pca, eps=args.eps_jacobian, scatter_labels=args.scatter_labels,
            max_prompts_for_overview=args.max_prompts_for_overview
        )
        log.info("Saved subspace CSV(s)/aggregate plots. Models: %s", ",".join(tags))

    # 3) Attention
    attn_sim_df = None
    tok_stats_df = None
    if args.run_attention_inspection:
        attention_heatmaps(items, ft, tokenizer, args.layer_index, args.device, outdir, args.prompt_span,
                           max_items=args.max_items_attn,
                           attn_keys=[k.strip() for k in args.attn_keys.split(",")] if args.attn_keys else None,
                           attn_keys_regex=args.attn_keys_regex)
        log.info("Skipped per-prompt attention grids (aggregated-only mode).")

    if args.run_attention_similarity:
        attn_sim_df = attention_similarity(items, base, ft, tokenizer, args.layer_index, args.device, outdir,
                                           args.prompt_span, max_items=max(100, args.max_items_attn),
                                           attn_keys=[k.strip() for k in args.attn_keys.split(",")] if args.attn_keys else None,
                                           attn_keys_regex=args.attn_keys_regex, metric_kl=True)
        log.info("Saved attention similarity CSV/plots.")

    if args.run_attention_token_stats:
        tok_stats_df = attention_token_stats(items, ft, tokenizer, args.layer_index, args.device, outdir,
                                             args.prompt_span, max_items=max(200, args.max_items_attn),
                                             attn_keys_regex=args.attn_keys_regex)
        log.info("Saved attention token stats CSV/plot.")

    # 4) Generation-level weight patching
    did_gen = False
    if args.run_weight_patching_eval:
        run_weight_patching_generation_eval(
            items=items, base=base, ft=ft, tokenizer=tokenizer,
            layer_index=args.layer_index, device=args.device, outdir=outdir,
            keys=keys, keys_regex=args.keys_regex, max_items=args.max_items_gen,
            max_new_tokens=args.decode_max_new_tokens, do_sample=args.decode_do_sample,
            temperature=args.decode_temperature, top_p=args.decode_top_p, top_k=args.decode_top_k
        )
        did_gen = True
        log.info("Saved generation-level weight patching CSV/plots.")

    # Report
    write_markdown_report(outdir, args.layer_index,
                          dod_pre=dod.get("pre"), dod_post=dod.get("post"),
                          subspace_df=subspace_df,
                          attn_sim_df=attn_sim_df,
                          token_stats_df=tok_stats_df,
                          did_gen_eval=did_gen,
                          models_for_subspace=[t.strip() for t in args.subspace_models.split(",") if t.strip()])


if __name__ == "__main__":
    main()
