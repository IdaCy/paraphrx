# (filename suggestion) analyze_paraphrase_normed.py

from __future__ import annotations

import argparse
import json
import logging
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

from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# ----------------
# Utils
# ----------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
    # x: [T,D] or [B,T,D]; mask: [T] or [B,T] (bool)
    if x.dim() == 2:
        m = mask.to(device=x.device, dtype=x.dtype)
        return (x * m.unsqueeze(-1)).sum(dim=0) / (m.sum() + 1e-8)
    elif x.dim() == 3:
        m = mask.to(device=x.device, dtype=x.dtype)
        return (x * m.unsqueeze(-1)).sum(dim=1) / (m.sum(dim=1, keepdim=True) + 1e-8)
    else:
        raise ValueError("masked_mean expects [T,D] or [B,T,D] tensors")

def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1.0 - cosine(a.unsqueeze(0), b.unsqueeze(0)).item())


# ----------------
# Data
# ----------------

@dataclass
class Item:
    prompt_count: int
    instruction_original: str
    paraphrases: Dict[str, str]
    input_text: str

    def all_prompt_variants(
        self,
        include_original: bool = True,
        keys: Optional[Iterable[str]] = None,
        regex: Optional[str] = None
    ) -> List[Tuple[str, str]]:
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
        items.append(Item(
            prompt_count=pc,
            instruction_original=instr,
            paraphrases=paraphrases,
            input_text=inp
        ))
    return items


# ----------------
# Tokenization & encoding
# ----------------

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

    return Encoded(
        input_ids=input_ids.to(device),
        attention_mask=attn_mask.to(device),
        prompt_mask=prompt_mask.to(device)
    )


# ----------------
# Model plumbing & hooks
# ----------------

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
                if getattr(dst, "bias", None) is not None and getattr(src, "bias", None) is not None:
                    dst.bias.data.copy_(src.bias.data)

    def swap_mlp_from(self, other: "BlockAccessor"):
        for name in ["up_proj", "gate_proj", "down_proj"]:
            src = getattr(other, name, None); dst = getattr(self, name, None)
            if src is not None and dst is not None:
                dst.weight.data.copy_(src.weight.data)
                if getattr(dst, "bias", None) is not None and getattr(src, "bias", None) is not None:
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

def capture_layer_resids(model, layer_index: int, device: str):
    layers = BlockAccessor(model, layer_index).layers
    block = layers[layer_index]

    resid_pre_list: List[torch.Tensor] = []
    resid_post_list: List[torch.Tensor] = []

    hooks = HookHandles()

    def pre_hook(module, inputs):
        resid_pre_list.append(inputs[0].detach().to("cpu"))
    def block_forward_hook(module, inputs, outputs):
        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        resid_post_list.append(hidden_states.detach().to("cpu"))

    hooks.add(block.register_forward_pre_hook(pre_hook))
    hooks.add(block.register_forward_hook(block_forward_hook))

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
                      output_attentions=False, return_dict=True)
        return CapturedActivations(resid_pre=resid_pre_list[:],
                                   resid_post=resid_post_list[:])

    run.hooks = hooks
    return run


# ----------------
# Jacobian probe (MLP)
# ----------------

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


# ----------------
# Representations
# ----------------

def prompt_representation_from_layer(model, layer_index: int, enc: Encoded, rep_at: str) -> torch.Tensor:
    runner = capture_layer_resids(model, layer_index, next(model.parameters()).device)
    try:
        caps = runner([enc])
        if rep_at == "pre":
            if not caps.resid_pre: raise RuntimeError("Failed to capture resid_pre.")
            x = caps.resid_pre[0][0]  # [T,D]
        else:
            if not caps.resid_post: raise RuntimeError("Failed to capture resid_post.")
            x = caps.resid_post[0][0]
        return masked_mean(x, enc.prompt_mask)  # [D]
    finally:
        runner.hooks.remove_all()


# ----------------
# Model builders
# ----------------

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
    hb_blk.swap_attention_from(ft_blk); hb_blk.swap_mlp_from(ft_blk)
    for m in [hybrid_attn, hybrid_mlp, hybrid_both]:
        m.eval()
    return hybrid_attn, hybrid_mlp, hybrid_both


# ----------------
# 1) Diff-of-diffs (rep level, cosine distance)
# ----------------

def diff_of_diffs_for_item(models: Dict[str, nn.Module], tokenizer, item: Item,
                           layer_index: int, device: str, prompt_span: str, rep_at: str) -> Dict[str, float]:
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
    # Δ = D_base - D_model (higher is better), matches your plots/text
    return {name: (baseD - v) if (not np.isnan(baseD) and not np.isnan(v)) else np.nan
            for name, v in avgD.items()}

def run_diff_of_diffs(items: List[Item], base: nn.Module, ft: nn.Module, tokenizer,
                      layer_index: int, device: str, outdir: Path, prompt_span: str) -> Dict[str, pd.DataFrame]:
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

        # concise summary plot
        fig, ax = plt.subplots(figsize=(7.2, 4))
        labels = ["FT", "HYB_ATTN", "HYB_MLP", "HYB_BOTH"]
        means = []
        Ns = []
        for col in labels:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            Ns.append(int(s.shape[0])); means.append(s.mean() if not s.empty else 0.0)
        ax.bar(labels, means)
        ax.set_title(f"Diff-of-diffs (avg across prompts) — layer {layer_index} ({rep_at})\nN={Ns}")
        ax.set_ylabel("Δ = D_base - D_model (cosine distance)")
        fig.tight_layout()
        fig.savefig(outdir / f"diff_of_diffs_layer{layer_index}_{rep_at}_summary.png", dpi=180)
        plt.close(fig)
        out[rep_at] = df
    return out


# ----------------
# 2) Paraphrase subspace (PCA) + Jacobian — with unit-norm option
# ----------------

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
    model_tag: str = "FT",
    rep_unit_norm: bool = True,
    keys: Optional[List[str]] = None,
    keys_regex: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - summary_df: per-prompt PCA & Jacobian metrics
      - norms_df: per-prompt raw norm stats (before unit-norm)
    """
    f_mlp = mlp_function_from_block(model, layer_index)
    norm_rows = []
    summary_rows = []

    rgx = re.compile(keys_regex) if keys_regex else None

    for item in items:
        H_list = []
        raw_norms = []

        for key, para in item.all_prompt_variants(include_original=True, keys=keys, regex=keys_regex):
            if key != "instruction_original" and rgx is not None and not rgx.search(key):
                continue
            enc = encode_prompt(tokenizer, build_prompt_text(para, item.input_text), device, prompt_span=prompt_span)
            runner = capture_layer_resids(model, layer_index, next(model.parameters()).device)
            try:
                caps = runner([enc])
                if not caps.resid_pre:
                    continue
                pre = caps.resid_pre[0][0]  # [T,D]
                h = masked_mean(pre, enc.prompt_mask)  # [D]
            finally:
                runner.hooks.remove_all()

            raw_norm = float(h.norm(p=2).item())
            raw_norms.append(raw_norm)

            if rep_unit_norm:
                h = F.normalize(h, dim=0, eps=1e-8)
            H_list.append(h.cpu().numpy().astype(np.float32))

        if len(H_list) < 3:
            continue

        # record raw norm stats per prompt (before any unit-norming)
        if raw_norms:
            raw_norms = np.array(raw_norms, dtype=np.float32)
            norm_rows.append({
                "model": model_tag,
                "prompt_count": item.prompt_count,
                "N": int(raw_norms.size),
                "norm_mean": float(raw_norms.mean()),
                "norm_std": float(raw_norms.std(ddof=1)) if raw_norms.size > 1 else 0.0,
                "norm_min": float(raw_norms.min()),
                "norm_max": float(raw_norms.max()),
                "norm_cv": float((raw_norms.std(ddof=1)/ (raw_norms.mean()+1e-12))) if raw_norms.size > 1 else 0.0,
            })

        H = np.stack(H_list, axis=0)  # [N, D]
        H_mean = H.mean(axis=0, keepdims=True)
        Hc = H - H_mean

        # PCA
        U, S, Vt = np.linalg.svd(Hc, full_matrices=False)
        var = (S ** 2)
        if var.sum() <= 0:
            continue
        varexpl = var / var.sum()

        # Jacobian norms along PCs vs random vs mean
        k = int(min(topk, Vt.shape[0]))
        PCs_k = Vt[:k]  # [k, D]
        model_dtype = next(model.parameters()).dtype
        PCs_t = torch.from_numpy(PCs_k).to(device).to(model_dtype)
        PCs_t = F.normalize(PCs_t, dim=-1)

        rand_dirs = torch.randn_like(PCs_t); rand_dirs = F.normalize(rand_dirs, dim=-1)
        H_mean_t = torch.from_numpy(H_mean.squeeze(0)).to(device).to(model_dtype)
        mean_dir = F.normalize(H_mean_t, dim=-1, eps=1e-8).unsqueeze(0)

        # Evaluate at each h (already unit if rep_unit_norm=True), then average over paraphrases
        h_tensor = torch.from_numpy(H).to(device).to(model_dtype)  # [N,D]
        norms_pcs_list, norms_rand_list, norms_mean_list = [], [], []
        for i in range(h_tensor.shape[0]):
            hi = h_tensor[i]
            norms_pcs_list.append(jacobian_directional_norms(f_mlp, hi, PCs_t, eps=eps).cpu().numpy())
            norms_rand_list.append(jacobian_directional_norms(f_mlp, hi, rand_dirs, eps=eps).cpu().numpy())
            norms_mean_list.append(jacobian_directional_norms(f_mlp, hi, mean_dir, eps=eps).cpu().numpy())

        norms_pcs = np.stack(norms_pcs_list, axis=0).mean(axis=0)   # [k]
        norms_rand = np.stack(norms_rand_list, axis=0).mean(axis=0) # [k]
        norms_mean = np.stack(norms_mean_list, axis=0).mean(axis=0) # [1]

        jac_PC_mean = float(norms_pcs.mean())
        jac_RANDOM_mean = float(norms_rand.mean())
        jac_MEAN = float(norms_mean.mean())

        summary_rows.append({
            "model": model_tag,
            "prompt_count": item.prompt_count,
            "rep_unit_norm": bool(rep_unit_norm),
            "N_paraphrases": int(H.shape[0]),
            "k_top": int(k),
            "pc1_var_expl": float(varexpl[0]) if len(varexpl) else np.nan,
            "cum_var_expl_topk": float(varexpl[:k].sum()) if len(varexpl) else np.nan,
            "jac_PC_mean": jac_PC_mean,
            "jac_RANDOM_mean": jac_RANDOM_mean,
            "jac_MEAN": jac_MEAN,
            "contraction_ratio_PC_over_RANDOM": float(jac_PC_mean / jac_RANDOM_mean) if jac_RANDOM_mean != 0 else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["model","prompt_count"])
    norms_df = pd.DataFrame(norm_rows).sort_values(["model","prompt_count"])
    return summary_df, norms_df

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
    rep_unit_norm: bool = True,
    keys: Optional[List[str]] = None,
    keys_regex: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dfs = []
    norms_all = []
    for tag, model in models.items():
        df, norms = paraphrase_pca_and_jacobian_for_model(
            items, model, tokenizer, layer_index, device, outdir, prompt_span,
            topk=topk, eps=eps, model_tag=tag, rep_unit_norm=rep_unit_norm,
            keys=keys, keys_regex=keys_regex
        )
        if not df.empty:
            dfs.append(df)
            df.to_csv(outdir / f"{tag}_paraphrase_subspace_jacobian_layer{layer_index}.csv", index=False)
        if not norms.empty:
            norms_all.append(norms)
            norms.to_csv(outdir / f"rep_norm_stats_{tag}_layer{layer_index}.csv", index=False)

    if not dfs:
        return pd.DataFrame(), (pd.DataFrame() if not norms_all else pd.concat(norms_all, ignore_index=True))

    cat = pd.concat(dfs, axis=0, ignore_index=True)
    cat.to_csv(outdir / f"ALL_paraphrase_subspace_jacobian_layer{layer_index}.csv", index=False)

    # concise across-model bars with 95% CI
    def _bar(metric: str, fname: str, ylim: Optional[Tuple[float,float]] = None):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        order = list(models.keys())
        means, stds, ns = [], [], []
        for m in order:
            s = pd.to_numeric(cat.loc[cat["model"]==m, metric], errors="coerce").dropna()
            means.append(s.mean() if not s.empty else 0.0)
            stds.append(s.std(ddof=1) if s.size > 1 else 0.0)
            ns.append(s.size)
        ci95 = [(1.96 * stds[i] / np.sqrt(ns[i])) if ns[i] > 1 else 0.0 for i in range(len(order))]
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
    _bar("pc1_var_expl", f"subspace_pc1_var_expl_layer{layer_index}.png")
    _bar("cum_var_expl_topk", f"subspace_cum_var_topk_layer{layer_index}.png")

    norms_cat = pd.concat(norms_all, ignore_index=True) if norms_all else pd.DataFrame()
    if not norms_cat.empty:
        norms_cat.to_csv(outdir / f"ALL_rep_norm_stats_layer{layer_index}.csv", index=False)

    return cat, norms_cat


# ----------------
# Markdown report
# ----------------

def write_markdown_report(
    outdir: Path,
    layer_index: int,
    dod_pre: Optional[pd.DataFrame],
    dod_post: Optional[pd.DataFrame],
    subspace_df: Optional[pd.DataFrame],
    norms_df: Optional[pd.DataFrame],
    models_for_subspace: List[str],
    rep_unit_norm: bool,
):
    md = []
    def add(s=""): md.append(s)

    add(f"# Paraphrase-robustness analysis — layer {layer_index}")
    add("")
    add("## 1) Representation diff-of-diffs (cosine distance)")
    for rep_at in ["pre", "post"]:
        csv = outdir / f"diff_of_diffs_layer{layer_index}_{rep_at}.csv"
        png = outdir / f"diff_of_diffs_layer{layer_index}_{rep_at}_summary.png"
        add(f"- **{rep_at}** CSV: `{csv.name}`" if csv.exists() else f"- **{rep_at}** CSV: (not run)")
        if png.exists(): add(f"  - Summary: `{png.name}`")
    add("Δ = D_base − D_model (higher is better).")
    add("")

    add("## 2) Paraphrase subspace (PCA) + MLP directional Jacobian")
    add(f"- Unit-norm per paraphrase representation: **{rep_unit_norm}**")
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
            f"subspace_pc1_var_expl_layer{layer_index}.png",
            f"subspace_cum_var_topk_layer{layer_index}.png",
        ]:
            if (outdir / f).exists():
                add(f"  - `{f}`")
    add("")
    norms_csv = outdir / f"ALL_rep_norm_stats_layer{layer_index}.csv"
    if norms_csv.exists():
        add("## 3) Raw representation-norm stats")
        add(f"- `{norms_csv.name}` (per-prompt mean/std/min/max/CV)")
    add("")

    (outdir / f"report_layer{layer_index}.md").write_text("\n".join(md), encoding="utf-8")


# ----------------
# CLI
# ----------------

def main():
    parser = argparse.ArgumentParser(description="Paraphrase robustness (norm-aware)")

    parser.add_argument("--instructions_json", type=str, required=True)

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

    parser.add_argument("--topk_pca", type=int, default=8)
    parser.add_argument("--eps_jacobian", type=float, default=1e-3)
    parser.add_argument("--rep_unit_norm", action="store_true")

    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--keys", type=str, default=None)
    parser.add_argument("--keys_regex", type=str, default=None)
    parser.add_argument("--subspace_models", type=str, default="BASE,FT,HYB_MLP")

    args = parser.parse_args()
    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("normed")

    items = load_instruction_json(args.instructions_json)
    if args.max_samples is not None and len(items) > args.max_samples:
        items = items[:args.max_samples]

    base, ft, tokenizer = build_model_and_tokenizer(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, args.device
    )

    hyb_attn, hyb_mlp, hyb_both = build_hybrid_models(base, ft, args.layer_index, args.device)
    MODELS = {"BASE": base, "FT": ft, "HYB_ATTN": hyb_attn, "HYB_MLP": hyb_mlp, "HYB_BOTH": hyb_both}

    keys = [k.strip()] if (args.keys and "," not in args.keys) else ([k.strip() for k in args.keys.split(",")] if args.keys else None)

    # 1) Diff-of-diffs (rep level)
    dod = {"pre": None, "post": None}
    if args.run_diff_of_diffs:
        dod = run_diff_of_diffs(items, base, ft, tokenizer, args.layer_index, args.device, outdir, args.prompt_span)
        log.info("Saved diff-of-diffs CSV/plots.")

    # 2) Subspace shrinkage + Jacobian (norm-aware)
    subspace_df = None
    norms_df = None
    if args.run_paraphrase_subspace:
        tags = [t.strip() for t in args.subspace_models.split(",") if t.strip() in MODELS]
        mm = {t: MODELS[t] for t in tags}
        subspace_df, norms_df = paraphrase_pca_and_jacobian_multi(
            items, mm, tokenizer, args.layer_index, args.device, outdir, args.prompt_span,
            topk=args.topk_pca, eps=args.eps_jacobian, rep_unit_norm=args.rep_unit_norm,
            keys=keys, keys_regex=args.keys_regex
        )
        log.info("Saved norm-aware subspace CSV(s)/summary plots. Models: %s", ",".join(tags))

    # Report
    write_markdown_report(outdir, args.layer_index,
                          dod_pre=dod.get("pre"), dod_post=dod.get("post"),
                          subspace_df=subspace_df, norms_df=norms_df,
                          models_for_subspace=[t.strip() for t in args.subspace_models.split(",") if t.strip()],
                          rep_unit_norm=args.rep_unit_norm)

if __name__ == "__main__":
    main()
