#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional LoRA host loading (not required here but kept for completeness)
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# Utils

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def format_eta(done: int, total: int, start_time: float) -> str:
    if done == 0: return "ETA --:--"
    elapsed = time.time() - start_time
    rate = done / max(elapsed, 1e-6)
    remain = (total - done) / max(rate, 1e-6)
    return f"ETA {int(remain//60):02d}:{int(remain%60):02d} (elapsed {int(elapsed//60):02d}:{int(elapsed%60):02d})"

@dataclass
class PromptSet:
    prompt_count: int
    instruction_original: str
    input_text: str
    paraphrases: List[Tuple[str,str]]  # (key, text)

def load_selection_jsonl(path: str | Path, max_prompts: Optional[int] = None) -> List[PromptSet]:
    items: List[PromptSet] = []
    n = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            paraphrases = [(p["key"], p["text"]) for p in obj.get("paraphrases", []) if isinstance(p, dict)]
            items.append(PromptSet(
                prompt_count=int(obj.get("prompt_count", n)),
                instruction_original=obj.get("instruction_original",""),
                input_text=obj.get("input","") or "",
                paraphrases=paraphrases
            ))
            n += 1
            if max_prompts and n >= max_prompts:
                break
    return items

def encode(tokenizer, text: str, device: torch.device):
    out = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
    return out["input_ids"].to(device), out["attention_mask"].to(device)

def mean_pool_tokens(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)
    s = (x * m.unsqueeze(-1)).sum(1)  # [1,D]
    c = m.sum(1).clamp(min=1.0)       # [1]
    return (s / c.unsqueeze(-1)).squeeze(0).to(torch.float32)  # [D]


# Model plumbing (Gemma-2 uses GeLU MLP)

class ResidualHook:
    def __init__(self, model, layer):
        self.layer = layer
        self.buffer = None
        self.mlp = self._get_mlp(model, layer)
        self.hook = self.mlp.register_forward_pre_hook(self._hook)
    def _get_mlp(self, model, i):
        for stem, leaf in [("model.model.layers","mlp"), ("model.layers","mlp"), ("transformer.h","mlp")]:
            try:
                base = model
                for part in stem.split("."): base = getattr(base, part)
                return getattr(base[i], leaf)
            except Exception: pass
        raise RuntimeError("Could not find mlp for residual hook")
    def _hook(self, module, inputs):
        self.buffer = inputs[0].detach()  # [B,T,D]
    def close(self):
        try: self.hook.remove()
        except Exception: pass

class MLPAccessor:
    def __init__(self, model, layer: int):
        self.model = model
        self.layer = layer
        self.mlp = self._get_mlp_module(model, layer)
        self.kind = self._detect_kind(self.mlp)
        p = next(self.mlp.parameters())
        self.device = p.device
        self.dtype = p.dtype
        self._act_fn = getattr(self.mlp, "act_fn", None) or getattr(self.mlp, "activation_fn", None)
        if self._act_fn is None:
            import torch.nn.functional as F
            self._act_fn = F.silu if self.kind == "swiglu" else F.gelu

    def _get_mlp_module(self, model, i: int):
        for stem, leaf in [("model.model.layers","mlp"), ("model.layers","mlp"), ("transformer.h","mlp")]:
            try:
                base = model
                for part in stem.split("."): base = getattr(base, part)
                return getattr(base[i], leaf)
            except Exception: pass
        raise RuntimeError(f"Could not locate MLP for layer {i}")

    def _detect_kind(self, mlp) -> str:
        if hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
            return "swiglu"
        if any(hasattr(mlp, n) for n in ["wi","fc_in","dense_h_to_4h"]):
            return "gelu"
        raise RuntimeError("Unknown MLP kind")

    @torch.no_grad()
    def UP(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu":
            return self.mlp.up_proj(h)
        # GeLU path
        if hasattr(self.mlp, "wi"): return self.mlp.wi(h)
        if hasattr(self.mlp, "fc_in"): return self.mlp.fc_in(h)
        if hasattr(self.mlp, "dense_h_to_4h"): return self.mlp.dense_h_to_4h(h)
        raise RuntimeError("Cannot find GeLU upstream linear")

    @torch.no_grad()
    def POST(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu":
            up = self.mlp.up_proj(h); gate = self._act_fn(self.mlp.gate_proj(h))
            return up * gate
        # GeLU path
        import torch.nn.functional as F
        if hasattr(self.mlp, "wi"): return F.gelu(self.mlp.wi(h))
        if hasattr(self.mlp, "fc_in"): return F.gelu(self.mlp.fc_in(h))
        if hasattr(self.mlp, "dense_h_to_4h"): return F.gelu(self.mlp.dense_h_to_4h(h))
        raise RuntimeError("Cannot find GeLU downstream POST")

def build_models_and_tokenizers(
    base_model_name_or_path: str,
    ft_model_name_or_path: Optional[str],
    ft_lora_adapter: Optional[str],
    device: torch.device,
):
    tok_base = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tok_base.pad_token is None: tok_base.pad_token = tok_base.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device).eval()

    if ft_lora_adapter is not None:
        host_name = ft_model_name_or_path or base_model_name_or_path
        ft_host = AutoModelForCausalLM.from_pretrained(host_name, torch_dtype=dtype).to(device).eval()
        if not _HAS_PEFT:
            raise RuntimeError("peft not available but --ft_lora_adapter provided.")
        ft = PeftModel.from_pretrained(ft_host, ft_lora_adapter).to(device).eval()
    else:
        if ft_model_name_or_path is None:
            raise ValueError("Provide --ft_model_name_or_path OR --ft_lora_adapter.")
        ft = AutoModelForCausalLM.from_pretrained(ft_model_name_or_path, torch_dtype=dtype).to(device).eval()

    tok_ft = AutoTokenizer.from_pretrained(ft_model_name_or_path or base_model_name_or_path, use_fast=True)
    if tok_ft.pad_token is None: tok_ft.pad_token = tok_ft.eos_token
    return (base, tok_base), (ft, tok_ft)


# Core experiment

def normalize(v: torch.Tensor) -> torch.Tensor:
    v = v.to(torch.float32)
    n = v.norm(p=2) + 1e-12
    return v / n

@torch.no_grad()
def collect_residuals(model, tokenizer, items: List[PromptSet], layer: int, device: torch.device,
                      max_paraphrases: Optional[int]) -> List[torch.Tensor]:
    """Return per-prompt residual matrices [N,D] (original + paraphrases) mean-pooled at the layer input."""
    hook = ResidualHook(model, layer)
    out: List[torch.Tensor] = []
    start = time.time()
    for idx, ps in enumerate(items):
        texts = [ps.instruction_original] + [t for _, t in ps.paraphrases]
        if max_paraphrases is not None:
            texts = texts[: 1 + max_paraphrases]
        if len(texts) < 1:
            continue
        rows = []
        for t in texts:
            input_ids, attention_mask = encode(tokenizer, t, device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            H = hook.buffer  # [1,T,D]
            rows.append(mean_pool_tokens(H, attention_mask))  # [D]
        out.append(torch.stack(rows, dim=0))  # [N,D]
        if (idx+1) % 20 == 0:
            logging.info("[collect] %d/%d — %s", idx+1, len(items), format_eta(idx+1, len(items), start))
    hook.close()
    return out

def principal_component(X: torch.Tensor) -> torch.Tensor:
    """First right-singular vector of centered X [N,D]."""
    Xc = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Xc.to(torch.float32), full_matrices=False)
    pc = Vh[0]
    return normalize(pc)

def dataset_baseline_norms(model, items_H: List[torch.Tensor], layer: int, scope: str) -> Dict[str, float]:
    """E[||x||], E[||UP(x)||], E[||POST(x)||] over dataset; scope in {'all_texts','original_only'}."""
    acc = MLPAccessor(model, layer)
    norms_b, norms_u, norms_p = [], [], []
    for H in items_H:
        rows = [H[0]] if scope == "original_only" else [r for r in H]
        for v in rows:
            v = v.to(torch.float32)
            norms_b.append(float(v.norm().item()))
            norms_u.append(float(acc.UP(v.unsqueeze(0)).squeeze(0).norm().item()))
            norms_p.append(float(acc.POST(v.unsqueeze(0)).squeeze(0).norm().item()))
    def mci(x):
        a = np.asarray(x, dtype=float)
        m = float(a.mean()) if a.size else float("nan")
        s = float(a.std(ddof=1)) if a.size > 1 else 0.0
        ci = 1.96 * s / max(1.0, math.sqrt(a.size))
        return m, ci, int(a.size)
    mb, cb, nb = mci(norms_b); mu, cu, nu = mci(norms_u); mp, cp, np_ = mci(norms_p)
    return {"before_mean": mb, "before_ci": cb, "before_n": nb,
            "up_mean": mu, "up_ci": cu, "up_n": nu,
            "nonlin_mean": mp, "nonlin_ci": cp, "nonlin_n": np_}

def prompt_baseline_norms(model, items_H: List[torch.Tensor], layer: int, scope: str) -> List[Dict[str, float]]:
    """Per-prompt baselines E_i||·|| for each step; aligned with items_H order."""
    acc = MLPAccessor(model, layer)
    out: List[Dict[str, float]] = []
    for H in items_H:
        rows = [H[0]] if scope == "original_only" else [r for r in H]
        b_vals, u_vals, p_vals = [], [], []
        for v in rows:
            v = v.to(torch.float32)
            b_vals.append(float(v.norm().item()))
            u_vals.append(float(acc.UP(v.unsqueeze(0)).squeeze(0).norm().item()))
            p_vals.append(float(acc.POST(v.unsqueeze(0)).squeeze(0).norm().item()))
        out.append({"before_mean": float(np.mean(b_vals)) if b_vals else float("nan"),
                    "up_mean": float(np.mean(u_vals)) if u_vals else float("nan"),
                    "nonlin_mean": float(np.mean(p_vals)) if p_vals else float("nan"),
                    "n": int(len(b_vals))})
    return out

def _build_unit_tracks(model, items_H: List[torch.Tensor], layer: int,
                       input_scale: float = 1.0, random_dirs: int = 8) -> Dict[str, List[List[float]]]:
    """
    Absolute unit-probe tracks:
      For each prompt family, build unit directions: MEAN, VAR(PC1), and RANDOM (averaged over K),
      then record [Before, After_UP, After_POST] = [1.0*s, ||UP(d)||, ||POST(d)||],
      where s=input_scale (default 1.0). 'Before' should be constant across directions.
    """
    acc = MLPAccessor(model, layer)
    tracks = {"MEAN": [], "VAR": [], "RANDOM": []}
    for Hmat in items_H:
        mean_dir = normalize(Hmat.mean(dim=0))     # unit
        var_dir  = principal_component(Hmat)       # unit

        rnd_vals = []
        for _ in range(random_dirs):
            r = torch.randn_like(mean_dir)
            d = normalize(r) * input_scale
            before = float(d.norm().item())                      # = input_scale
            u = acc.UP(d.unsqueeze(0)).squeeze(0).norm().item()
            p = acc.POST(d.unsqueeze(0)).squeeze(0).norm().item()
            rnd_vals.append([before, float(u), float(p)])
        rnd_mean = np.mean(np.asarray(rnd_vals), axis=0).tolist()

        for name, vec in [("MEAN", mean_dir), ("VAR", var_dir)]:
            d = vec * input_scale
            before = float(d.norm().item())
            u = acc.UP(d.unsqueeze(0)).squeeze(0).norm().item()
            p = acc.POST(d.unsqueeze(0)).squeeze(0).norm().item()
            tracks[name].append([before, float(u), float(p)])

        tracks["RANDOM"].append(rnd_mean)
    return tracks

def _aggregate_tracks(rows: Dict[str, List[List[float]]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, triples in rows.items():
        A = np.asarray(triples, dtype=float)  # [num_prompts, 3]
        means = A.mean(axis=0)
        stds  = A.std(axis=0, ddof=1) if A.shape[0] > 1 else np.zeros_like(means)
        cis   = 1.96 * stds / max(1.0, math.sqrt(A.shape[0]))
        out[name] = {"before_mean": float(means[0]), "before_ci": float(cis[0]),
                     "up_mean": float(means[1]), "up_ci": float(cis[1]),
                     "nonlin_mean": float(means[2]), "nonlin_ci": float(cis[2]),
                     "n": int(A.shape[0]),
                     "ratio_up_over_before": float(means[1] / max(means[0], 1e-12)),
                     "ratio_nonlin_over_up": float(means[2] / max(means[1], 1e-12))}
    return out


def _calibrated_dataset_relative_rows(tracks: Dict[str, List[List[float]]],
                                      dataset_baseline: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Make per-prompt rows for RELATIVE (dataset) with calibration:
      - multiply each probe triple by s = dataset_baseline['before_mean'] (so Before == baseline)
      - then divide each element by the matching dataset mean.
      => Before_rel is identically 1.0 (up to fp error).
    """
    b_mean = float(dataset_baseline["before_mean"])
    u_mean = float(dataset_baseline["up_mean"])
    p_mean = float(dataset_baseline["nonlin_mean"])
    rows: List[Dict[str, Any]] = []
    for kind, per_prompt in tracks.items():
        for i, (before, up, post) in enumerate(per_prompt):
            # calibrate to baseline
            cb, cu, cp = before * b_mean, up * b_mean, post * b_mean
            rows.append({
                "dir_type": kind,
                "prompt_index": i,
                "before_rel": cb / max(b_mean, 1e-12),
                "after_up_rel": cu / max(u_mean, 1e-12),
                "after_nonlin_rel": cp / max(p_mean, 1e-12),
            })
    return rows

def _calibrated_prompt_relative_rows(tracks: Dict[str, List[List[float]]],
                                     prompt_baselines: List[Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Make per-prompt rows for RELATIVE (per-prompt) with calibration:
      - for prompt i, multiply triple by s_i = prompt_baselines[i]['before_mean']
      - divide by prompt-baseline means (per step).
      => Before_rel_prompt is identically 1.0.
    """
    rows: List[Dict[str, Any]] = []
    for kind, per_prompt in tracks.items():
        for i, (before, up, post) in enumerate(per_prompt):
            b_mean = float(prompt_baselines[i]["before_mean"])
            u_mean = float(prompt_baselines[i]["up_mean"])
            p_mean = float(prompt_baselines[i]["nonlin_mean"])
            cb, cu, cp = before * b_mean, up * b_mean, post * b_mean
            rows.append({
                "dir_type": kind,
                "prompt_index": i,
                "before_rel_prompt": cb / max(b_mean, 1e-12),
                "after_up_rel_prompt": cu / max(u_mean, 1e-12),
                "after_nonlin_rel_prompt": cp / max(p_mean, 1e-12),
            })
    return rows

def _aggregate_rel_rows(rows: List[Dict[str, Any]], which: str) -> Dict[str, Dict[str, float]]:
    """
    which in {'dataset','prompt'} chooses column names: *_rel vs *_rel_prompt.
    Returns dict like aggregate() with means + CIs per dir_type.
    """
    if which == "dataset":
        cb, cu, cp = "before_rel", "after_up_rel", "after_nonlin_rel"
    else:
        cb, cu, cp = "before_rel_prompt", "after_up_rel_prompt", "after_nonlin_rel_prompt"
    out: Dict[str, Dict[str, float]] = {}
    for name in ["MEAN","VAR","RANDOM"]:
        sub = [r for r in rows if r["dir_type"] == name]
        if not sub:
            out[name] = {"before_mean": float("nan"), "before_ci": 0.0,
                         "up_mean": float("nan"), "up_ci": 0.0,
                         "nonlin_mean": float("nan"), "nonlin_ci": 0.0}
            continue
        a_b = np.asarray([r[cb] for r in sub], dtype=float)
        a_u = np.asarray([r[cu] for r in sub], dtype=float)
        a_p = np.asarray([r[cp] for r in sub], dtype=float)
        def mci(a):
            m = float(a.mean())
            s = float(a.std(ddof=1)) if a.size > 1 else 0.0
            ci = 1.96 * s / max(1.0, math.sqrt(a.size))
            return m, ci
        bm, bci = mci(a_b); um, uci = mci(a_u); pm, pci = mci(a_p)
        out[name] = {"before_mean": bm, "before_ci": bci,
                     "up_mean": um, "up_ci": uci,
                     "nonlin_mean": pm, "nonlin_ci": pci}
    return out


# Plot helpers (gray + forest green)

def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0,2,4))

def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def _lighten(hex_color: str, factor: float):
    r,g,b = _hex_to_rgb(hex_color)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return _rgb_to_hex((r,g,b))

BASE_GRAY_SOLID = "#777777"
BASE_GRAY_DASH  = "#999999"
BASE_GRAY_DOT   = "#BBBBBB"
FT_GREEN        = "#0b5d1e"
FT_SOLID        = _lighten(FT_GREEN, 0.30)
FT_DASH         = _lighten(FT_GREEN, 0.45)
FT_DOT          = _lighten(FT_GREEN, 0.55)

def _series_from_agg(agg, name: str):
    return [agg[name]["before_mean"], agg[name]["up_mean"], agg[name]["nonlin_mean"]]

def plot_overlay_abs(agg_b, agg_f, out_path: Path, title: str):
    labels = ["Before","After UP","After Nonlinearity"]
    x = [0,1,2]
    plt.figure(figsize=(8.6, 5.0))
    if "MEAN" in agg_b: plt.plot(x, _series_from_agg(agg_b,"MEAN"), marker="o", linewidth=2.2, label="BASE MEAN", color=BASE_GRAY_SOLID)
    if "VAR"  in agg_b: plt.plot(x, _series_from_agg(agg_b,"VAR"),  marker="o", linewidth=1.9, label="BASE VAR",  color=BASE_GRAY_DASH, linestyle="--")
    if "RANDOM" in agg_b: plt.plot(x, _series_from_agg(agg_b,"RANDOM"), marker="o", linewidth=1.8, label="BASE RANDOM", color=BASE_GRAY_DOT, linestyle=":")

    if "MEAN" in agg_f: plt.plot(x, _series_from_agg(agg_f,"MEAN"), marker="o", linewidth=2.2, label="FT MEAN", color=FT_SOLID)
    if "VAR"  in agg_f: plt.plot(x, _series_from_agg(agg_f,"VAR"),  marker="o", linewidth=1.9, label="FT VAR",  color=FT_DASH, linestyle="--")
    if "RANDOM" in agg_f: plt.plot(x, _series_from_agg(agg_f,"RANDOM"), marker="o", linewidth=1.8, label="FT RANDOM", color=FT_DOT, linestyle=":")

    plt.xticks(x, labels); plt.ylabel("Vector norm"); plt.title(title)
    plt.grid(alpha=0.15, linestyle="--", linewidth=0.8); plt.legend(ncol=2)
    plt.tight_layout(); plt.savefig(out_path, dpi=170); plt.close()

def plot_overlay_rel(agg_b, agg_f, out_path: Path, title: str, ylab: str = "Relative norm"):
    labels = ["Before (rel)","After UP (rel)","After Nonlin (rel)"]
    x = [0,1,2]
    plt.figure(figsize=(8.6, 5.0))
    if "MEAN" in agg_b: plt.plot(x, _series_from_agg(agg_b,"MEAN"), marker="o", linewidth=2.2, label="BASE MEAN", color=BASE_GRAY_SOLID)
    if "VAR"  in agg_b: plt.plot(x, _series_from_agg(agg_b,"VAR"),  marker="o", linewidth=1.9, label="BASE VAR",  color=BASE_GRAY_DASH, linestyle="--")
    if "RANDOM" in agg_b: plt.plot(x, _series_from_agg(agg_b,"RANDOM"), marker="o", linewidth=1.8, label="BASE RANDOM", color=BASE_GRAY_DOT, linestyle=":")

    if "MEAN" in agg_f: plt.plot(x, _series_from_agg(agg_f,"MEAN"), marker="o", linewidth=2.2, label="FT MEAN", color=FT_SOLID)
    if "VAR"  in agg_f: plt.plot(x, _series_from_agg(agg_f,"VAR"),  marker="o", linewidth=1.9, label="FT VAR",  color=FT_DASH, linestyle="--")
    if "RANDOM" in agg_f: plt.plot(x, _series_from_agg(agg_f,"RANDOM"), marker="o", linewidth=1.8, label="FT RANDOM", color=FT_DOT, linestyle=":")

    plt.xticks(x, labels); plt.ylabel(ylab); plt.title(title)
    plt.grid(alpha=0.15, linestyle="--", linewidth=0.8); plt.legend(ncol=2)
    plt.tight_layout(); plt.savefig(out_path, dpi=170); plt.close()

def _suppression_bar(abs_agg_base, abs_agg_ft, out_path: Path):
    labels = ["MEAN","VAR","RANDOM"]; x = np.arange(len(labels)); w = 0.35
    base_vals = [abs_agg_base[k]["ratio_nonlin_over_up"] for k in labels]
    ft_vals   = [abs_agg_ft[k]["ratio_nonlin_over_up"] for k in labels]
    plt.figure(figsize=(7.6,4.6))
    plt.bar(x-w/2, base_vals, width=w, label="BASE", color="#6e6e6e", edgecolor="black")
    plt.bar(x+w/2, ft_vals,   width=w, label="FT",   color="#0b5d1e")
    plt.xticks(x, labels); plt.axhline(0, color="#6e6e6e", linewidth=0.9)
    plt.ylabel("Nonlin/UP (smaller = stronger cut)")
    plt.title("Suppression ratio at nonlinearity")
    plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=170); plt.close()


# Main

def main():
    ap = argparse.ArgumentParser(description="UP+Nonlinearity attenuation: absolute + calibrated relatives")
    ap.add_argument("--selection_jsonl", type=str, required=True)
    ap.add_argument("--base_model_name_or_path", type=str, required=True)
    ap.add_argument("--ft_model_name_or_path", type=str, default=None)
    ap.add_argument("--ft_lora_adapter", type=str, default=None)
    ap.add_argument("--layer_index", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--max_paraphrases", type=int, default=None)
    ap.add_argument("--random_dirs", type=int, default=8)
    ap.add_argument("--input_scale", type=float, default=1.0, help="unit-probe scale for ABSOLUTE view (kept at 1.0)")
    ap.add_argument("--baseline_scope", type=str, default="all_texts", choices=["all_texts","original_only"])
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    device = torch.device(args.device)
    logging.info("Loading selection: %s", args.selection_jsonl)
    items = load_selection_jsonl(args.selection_jsonl, max_prompts=args.max_prompts)
    logging.info("Loaded %d prompt groups", len(items))

    (base, tok_b), (ft, tok_f) = build_models_and_tokenizers(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, device
    )
    L = args.layer_index
    logging.info("Collecting residuals (BASE) at layer %d", L)
    H_base = collect_residuals(base, tok_b, items, L, device, args.max_paraphrases)
    logging.info("Collecting residuals (FT) at layer %d", L)
    H_ft   = collect_residuals(ft,   tok_f, items, L, device, args.max_paraphrases)

    # Baselines
    logging.info("Computing dataset baselines (scope=%s): BASE", args.baseline_scope)
    bl_base = dataset_baseline_norms(base, H_base, L, args.baseline_scope)
    logging.info("Computing dataset baselines (scope=%s): FT", args.baseline_scope)
    bl_ft   = dataset_baseline_norms(ft,   H_ft,   L, args.baseline_scope)

    logging.info("Computing per-prompt baselines (scope=%s): BASE", args.baseline_scope)
    pbl_base = prompt_baseline_norms(base, H_base, L, args.baseline_scope)
    logging.info("Computing per-prompt baselines (scope=%s): FT", args.baseline_scope)
    pbl_ft   = prompt_baseline_norms(ft,   H_ft,   L, args.baseline_scope)

    # Save baselines
    (Path(outdir)/"baselines.json").write_text(json.dumps({"BASE": bl_base, "FT": bl_ft, "scope": args.baseline_scope}, indent=2), encoding="utf-8")
    import pandas as pd
    pd.DataFrame(pbl_base).assign(model="BASE", prompt_index=lambda df: df.index).to_csv(Path(outdir)/f"prompt_baselines_BASE_layer{L}.csv", index=False)
    pd.DataFrame(pbl_ft).assign(model="FT",   prompt_index=lambda df: df.index).to_csv(Path(outdir)/f"prompt_baselines_FT_layer{L}.csv", index=False)

    # Absolute unit-probe tracks (input_scale kept at 1.0 to preserve the unit-probe convention)
    logging.info("Building absolute unit-probe tracks (BASE)")
    tracks_base_abs = _build_unit_tracks(base, H_base, L, input_scale=args.input_scale, random_dirs=args.random_dirs)
    logging.info("Building absolute unit-probe tracks (FT)")
    tracks_ft_abs   = _build_unit_tracks(ft,   H_ft,   L, input_scale=args.input_scale, random_dirs=args.random_dirs)

    abs_base_agg = _aggregate_tracks(tracks_base_abs)
    abs_ft_agg   = _aggregate_tracks(tracks_ft_abs)

    # Calibrated RELATIVE rows & aggregates
    # Dataset-relative (calibrated)
    rows_base_rel_ds = _calibrated_dataset_relative_rows(tracks_base_abs, bl_base)
    rows_ft_rel_ds   = _calibrated_dataset_relative_rows(tracks_ft_abs,   bl_ft)
    agg_base_rel_ds  = _aggregate_rel_rows(rows_base_rel_ds, which="dataset")
    agg_ft_rel_ds    = _aggregate_rel_rows(rows_ft_rel_ds,   which="dataset")

    # Prompt-relative (calibrated)
    rows_base_rel_pr = _calibrated_prompt_relative_rows(tracks_base_abs, pbl_base)
    rows_ft_rel_pr   = _calibrated_prompt_relative_rows(tracks_ft_abs,   pbl_ft)
    agg_base_rel_pr  = _aggregate_rel_rows(rows_base_rel_pr, which="prompt")
    agg_ft_rel_pr    = _aggregate_rel_rows(rows_ft_rel_pr,   which="prompt")

    # Persist per-prompt CSV with absolute + both calibrated relatives
    csv_rows = []
    # absolute per-prompt rows
    for model_name, tracks in [("BASE", tracks_base_abs), ("FT", tracks_ft_abs)]:
        for kind, per_prompt in tracks.items():
            for i, (before, up, post) in enumerate(per_prompt):
                csv_rows.append({"model": model_name, "dir_type": kind, "prompt_index": i,
                                 "before": before, "after_up": up, "after_nonlin": post})
    # attach calibrated dataset-relative
    for model_name, rel_rows in [("BASE", rows_base_rel_ds), ("FT", rows_ft_rel_ds)]:
        for r in rel_rows:
            csv_rows.append({"model": model_name, "dir_type": r["dir_type"], "prompt_index": r["prompt_index"],
                             "before_rel_dataset": r["before_rel"],
                             "after_up_rel_dataset": r["after_up_rel"],
                             "after_nonlin_rel_dataset": r["after_nonlin_rel"]})
    # attach calibrated prompt-relative
    for model_name, rel_rows in [("BASE", rows_base_rel_pr), ("FT", rows_ft_rel_pr)]:
        for r in rel_rows:
            csv_rows.append({"model": model_name, "dir_type": r["dir_type"], "prompt_index": r["prompt_index"],
                             "before_rel_prompt": r["before_rel_prompt"],
                             "after_up_rel_prompt": r["after_up_rel_prompt"],
                             "after_nonlin_rel_prompt": r["after_nonlin_rel_prompt"]})
    # consolidate per (model, dir_type, prompt_index)
    # (so we don't create separate lines, but merge columns)
    by_key: Dict[Tuple[str,str,int], Dict[str,Any]] = {}
    for r in csv_rows:
        k = (r["model"], r["dir_type"], r["prompt_index"])
        by_key.setdefault(k, {"model":k[0], "dir_type":k[1], "prompt_index":k[2]})
        by_key[k].update(r)
    keys = ["model","dir_type","prompt_index",
            "before","after_up","after_nonlin",
            "before_rel_dataset","after_up_rel_dataset","after_nonlin_rel_dataset",
            "before_rel_prompt","after_up_rel_prompt","after_nonlin_rel_prompt"]
    with open(Path(outdir)/f"gate_norm_tracks_layer{L}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows([by_key[k] for k in sorted(by_key)])

    # Markdown tables
    def _md_abs(agg, model):
        lines = []
        lines.append(f"### Absolute norms (layer {L}) — {model}")
        lines.append("| Direction | Before | After UP | After Nonlinearity | UP/Before | Nonlin/UP |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name in ["MEAN","VAR","RANDOM"]:
            s = agg[name]
            lines.append(f"| {name} | {s['before_mean']:.4f} ± {s['before_ci']:.4f} | {s['up_mean']:.4f} ± {s['up_ci']:.4f} | {s['nonlin_mean']:.4f} ± {s['nonlin_ci']:.4f} | {s['ratio_up_over_before']:.3f} | {s['ratio_nonlin_over_up']:.3f} |")
        return "\n".join(lines)

    def _md_rel(title, agg_b, agg_f):
        lines = []
        lines.append(f"### {title} (layer {L})")
        lines.append("| Model | Direction | Before_rel | Up_rel | Nonlin_rel |")
        lines.append("|---|---|---:|---:|---:|")
        for model_name, agg in [("BASE", agg_b), ("FT", agg_f)]:
            for name in ["MEAN","VAR","RANDOM"]:
                s = agg[name]
                lines.append(f"| {model_name} | {name} | {s['before_mean']:.4f} ± {s['before_ci']:.4f} | {s['up_mean']:.4f} ± {s['up_ci']:.4f} | {s['nonlin_mean']:.4f} ± {s['nonlin_ci']:.4f} |")
        return "\n".join(lines)

    (Path(outdir)/f"gate_norm_abs_BASE_layer{L}.md").write_text(_md_abs(abs_base_agg, "BASE"), encoding="utf-8")
    (Path(outdir)/f"gate_norm_abs_FT_layer{L}.md").write_text(_md_abs(abs_ft_agg,   "FT"),   encoding="utf-8")

    # Dataset-relative (calibrated): Before_rel must be ~1.000 across directions
    (Path(outdir)/f"gate_norm_rel_dataset_layer{L}.md").write_text(
        _md_rel("Relative (dataset, calibrated)", agg_base_rel_ds, agg_ft_rel_ds), encoding="utf-8"
    )
    # Prompt-relative (calibrated): Before_rel_prompt must be ~1.000 across directions
    (Path(outdir)/f"gate_norm_rel_prompt_layer{L}.md").write_text(
        _md_rel("Relative (per-prompt, calibrated)", agg_base_rel_pr, agg_ft_rel_pr), encoding="utf-8"
    )

    # Baselines table
    lines = []
    lines.append(f"### Baselines (scope={args.baseline_scope}, layer {L})")
    lines.append("| Model | E[||x||] | E[||UP(x)||] | E[||POST(x)||] | n |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| BASE | {bl_base['before_mean']:.4f} ± {bl_base['before_ci']:.4f} | {bl_base['up_mean']:.4f} ± {bl_base['up_ci']:.4f} | {bl_base['nonlin_mean']:.4f} ± {bl_base['nonlin_ci']:.4f} | {bl_base['before_n']} |")
    lines.append(f"| FT   | {bl_ft['before_mean']:.4f} ± {bl_ft['before_ci']:.4f} | {bl_ft['up_mean']:.4f} ± {bl_ft['up_ci']:.4f} | {bl_ft['nonlin_mean']:.4f} ± {bl_ft['nonlin_ci']:.4f} | {bl_ft['before_n']} |")
    (Path(outdir)/f"baselines_layer{L}.md").write_text("\n".join(lines), encoding="utf-8")

    # Plots
    # Absolute per-model + overlay
    plot_overlay_abs(abs_base_agg, abs_ft_agg, Path(outdir)/"norm_lines_overlay_abs.png",
                     title=f"UP + Nonlinearity (absolute) — layer {L}")
    # Dataset-relative (calibrated)
    plot_overlay_rel(agg_base_rel_ds, agg_ft_rel_ds, Path(outdir)/"norm_lines_overlay_rel_dataset.png",
                     title=f"UP + Nonlinearity (relative to dataset, calibrated) — layer {L}",
                     ylab="Relative to dataset baseline")
    # Prompt-relative (calibrated)
    plot_overlay_rel(agg_base_rel_pr, agg_ft_rel_pr, Path(outdir)/"norm_lines_overlay_rel_prompt.png",
                     title=f"UP + Nonlinearity (relative to per-prompt, calibrated) — layer {L}",
                     ylab="Relative to per-prompt baseline")

    # Suppression bars (Nonlin/UP) from absolute
    _suppression_bar(abs_base_agg, abs_ft_agg, Path(outdir)/"suppression_ratio_nonlin_over_up.png")

    logging.info("All done. Outputs in %s", outdir)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
