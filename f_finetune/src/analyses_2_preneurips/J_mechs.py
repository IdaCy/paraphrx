#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional LoRA
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# Utils

def set_seed(seed: int = 42):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def format_eta(done: int, total: int, start_time: float) -> str:
    if done == 0: return "ETA --:-- (elapsed 00:00)"
    elapsed = time.time() - start_time
    rate = done / max(elapsed, 1e-6)
    remain = (total - done) / max(rate, 1e-6)
    return f"ETA {int(remain//60):02d}:{int(remain%60):02d} (elapsed {int(elapsed//60):02d}:{int(elapsed%60):02d})"

def batched(iterable: Iterable[Any], n: int) -> Iterable[List[Any]]:
    """Yield lists of length <= n from iterable."""
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch; batch = []
    if batch: yield batch

@dataclass
class PromptSet:
    prompt_count: int
    instruction_original: str
    input_text: str
    paraphrases: List[Tuple[str, str]]  # (key, text)

def load_selection_jsonl(path: str | Path, max_prompts: Optional[int] = None,
                         min_content_score: Optional[int] = None,
                         allow_instruct_types: Optional[set] = None) -> List[PromptSet]:
    """Selection JSONL format: one object per prompt, with "paraphrases":[{"key","text",...}]."""
    items: List[PromptSet] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            raw = obj.get("paraphrases", [])
            para: List[Tuple[str,str]] = []
            for p in raw:
                if not isinstance(p, dict): continue
                # Optional filtering by content score and type
                if min_content_score is not None and p.get("paraphrase_content_score", 5) < min_content_score:
                    continue
                if allow_instruct_types is not None and p.get("key") not in allow_instruct_types \
                   and p.get("instruct_type") not in allow_instruct_types:
                    continue
                key = p.get("key") or p.get("instruct_type") or "PARA"
                para.append((key, p.get("text") or p.get("paraphrase") or ""))
            items.append(PromptSet(
                prompt_count=int(obj["prompt_count"]),
                instruction_original=obj.get("instruction_original",""),
                input_text=obj.get("input","") or "",
                paraphrases=para
            ))
            if max_prompts is not None and len(items) >= max_prompts:
                break
    return items

def _parse_token_slice(spec: str | None) -> Tuple[str, float]:
    # "head:0.7" / "tail:0.5" or "all"
    if spec is None or spec.lower() == "all":
        return ("all", 1.0)
    try:
        side, ratio = spec.split(":")
        return (side, float(ratio))
    except Exception:
        return ("all", 1.0)

def encode(tokenizer, texts: List[str], device: torch.device):
    out = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    return out["input_ids"].to(device), out["attention_mask"].to(device)

def _slice_mask(mask: torch.Tensor, side: str, ratio: float) -> torch.Tensor:
    """Keep only a prefix/suffix/all of valid tokens in mask shape [B, T]."""
    if side == "all" or ratio >= 1.0: return mask
    B, T = mask.shape
    m2 = torch.zeros_like(mask)
    valid_counts = mask.sum(dim=1)
    for b in range(B):
        n = int(valid_counts[b].item())
        if n <= 0: continue
        k = max(1, int(round(n * ratio)))
        if side == "head":
            m2[b, :k] = mask[b, :k]
        else:
            m2[b, n-k:n] = mask[b, n-k:n]
    return m2

def mean_pool_tokens(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)
    s = (x * m.unsqueeze(-1)).sum(1)  # [B,D] summed over tokens
    c = m.sum(1).clamp(min=1.0).unsqueeze(-1)
    return (s / c).to(torch.float32)  # [B,D]

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

def symmetric_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Symmetric KL on same-shape logits [B,V]."""
    p = F.log_softmax(p_logits, dim=-1); q = F.log_softmax(q_logits, dim=-1)
    P = p.exp(); Q = q.exp()
    kl_pq = (P * (p - q)).sum(dim=-1)
    kl_qp = (Q * (q - p)).sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp)

# Model plumbing

class MLPAccessor:
    """Access UP/GATE/POST/DOWN given a transformer layer MLP module."""
    def __init__(self, model, layer: int, gate_mode: str = "silu"):
        self.model = model; self.layer = layer
        self.mlp = self._get_mlp_module(model, layer)
        self.kind = self._detect_kind(self.mlp)
        # If it's a SwiGLU MLP, default to SiLU gating; else GeLU.
        if self.kind == "swiglu" and gate_mode == "silu":
            self.gate_mode = "silu"
        elif self.kind != "swiglu" and gate_mode != "gelu":
            self.gate_mode = "gelu"
        else:
            self.gate_mode = gate_mode
        p = next(self.mlp.parameters())
        self.device = p.device; self.dtype = p.dtype

    def _get_mlp_module(self, model, i: int):
        for stem, leaf in [("model.model.layers","mlp"), ("model.layers","mlp"), ("transformer.h","mlp")]:
            try:
                base = model
                for part in stem.split("."): base = getattr(base, part)
                return getattr(base[i], leaf)
            except Exception: pass
        raise RuntimeError(f"Could not locate MLP for layer {i}")

    def _detect_kind(self, mlp) -> str:
        if hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"): return "swiglu"
        if any(hasattr(mlp, n) for n in ["wi","fc_in","dense_h_to_4h"]): return "gelu"
        return "unknown"

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "swiglu":
            # gemma/llama-style: SiLU gate by default
            if   self.gate_mode == "silu":    return F.silu(x)
            elif self.gate_mode == "gelu":    return F.gelu(x)
            elif self.gate_mode == "sigmoid": return torch.sigmoid(x)
            elif self.gate_mode == "relu":    return F.relu(x)
            elif self.gate_mode == "none":    return torch.ones_like(x)
            else:                              return F.silu(x)
        else:
            return F.gelu(x)

    @torch.no_grad()
    def UP(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu": return self.mlp.up_proj(h)
        if hasattr(self.mlp, "wi"): return self.mlp.wi(h)
        if hasattr(self.mlp, "fc_in"): return self.mlp.fc_in(h)
        if hasattr(self.mlp, "dense_h_to_4h"): return self.mlp.dense_h_to_4h(h)
        raise RuntimeError("Unknown upstream for GeLU MLP")

    @torch.no_grad()
    def GATE_PRE(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu": return self.mlp.gate_proj(h)
        # GeLU single branch
        if hasattr(self.mlp, "wi"): return self.mlp.wi(h)
        if hasattr(self.mlp, "fc_in"): return self.mlp.fc_in(h)
        if hasattr(self.mlp, "dense_h_to_4h"): return self.mlp.dense_h_to_4h(h)
        raise RuntimeError("Unknown gate_pre for GeLU MLP")

    @torch.no_grad()
    def GATE_ACT(self, h: torch.Tensor) -> torch.Tensor:
        if self.kind == "swiglu":
            return self._act(self.mlp.gate_proj(h.to(self.device, self.dtype)))
        else:
            pre = self.GATE_PRE(h)
            return F.gelu(pre)

    @torch.no_grad()
    def POST(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu":
            up = self.mlp.up_proj(h); gate = self._act(self.mlp.gate_proj(h))
            return up * gate
        else:
            if hasattr(self.mlp, "wi"): pre = self.mlp.wi(h)
            elif hasattr(self.mlp, "fc_in"): pre = self.mlp.fc_in(h)
            elif hasattr(self.mlp, "dense_h_to_4h"): pre = self.mlp.dense_h_to_4h(h)
            else: raise RuntimeError("Unknown GeLU upstream")
            return F.gelu(pre)

    @torch.no_grad()
    def DOWN(self, post: torch.Tensor) -> torch.Tensor:
        post = post.to(self.device, self.dtype)
        if hasattr(self.mlp, "down_proj"): return self.mlp.down_proj(post)
        if hasattr(self.mlp, "wo"): return self.mlp.wo(post)
        if hasattr(self.mlp, "fc_out"): return self.mlp.fc_out(post)
        if hasattr(self.mlp, "dense_4h_to_h"): return self.mlp.dense_4h_to_h(post)
        raise RuntimeError("Unknown downstream for GeLU MLP")

class ResidualHook:
    def __init__(self, model, layer):
        self.model = model; self.layer = layer; self.buffer = None
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

def build_models_and_tokenizers(
    base_model_name_or_path: str,
    ft_model_name_or_path: Optional[str],
    ft_lora_adapter: Optional[str],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
):
    tokenizer_base = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tokenizer_base.pad_token is None: tokenizer_base.pad_token = tokenizer_base.eos_token
    if dtype is None: dtype = torch.float16 if device.type == "cuda" else torch.float32

    base = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device).eval()

    if ft_lora_adapter is not None:
        host_name = ft_model_name_or_path or base_model_name_or_path
        ft_host = AutoModelForCausalLM.from_pretrained(host_name, torch_dtype=dtype).to(device).eval()
        if not _HAS_PEFT:
            raise RuntimeError("peft not available but --ft_lora_adapter provided.")
        ft = PeftModel.from_pretrained(ft_host, ft_lora_adapter).to(device).eval()
    else:
        if ft_model_name_or_path is None:
            raise ValueError("Provide --ft_model_name_or_path or --ft_lora_adapter.")
        ft = AutoModelForCausalLM.from_pretrained(ft_model_name_or_path, torch_dtype=dtype).to(device).eval()

    tokenizer_ft = AutoTokenizer.from_pretrained(ft_model_name_or_path or base_model_name_or_path, use_fast=True)
    if tokenizer_ft.pad_token is None: tokenizer_ft.pad_token = tokenizer_ft.eos_token
    return (base, tokenizer_base), (ft, tokenizer_ft)

# Station capture

def capture_stations(model, tokenizer, items: List[PromptSet], layer: int,
                     device: torch.device, token_slice: Tuple[str,float],
                     max_prompts: Optional[int], max_paraphrases: Optional[int]) -> Dict[str, Any]:
    """Return dict with per-prompt station matrices and logits of next token."""
    accessor = MLPAccessor(model, layer, gate_mode="silu")
    res_hook = ResidualHook(model, layer)
    results: Dict[str, Any] = {"per_prompt": [], "layer": layer}
    start = time.time()

    for idx, ps in enumerate(items):
        texts = [ps.instruction_original] + [t for _, t in ps.paraphrases]
        if max_paraphrases is not None: texts = texts[: 1 + max_paraphrases]
        if len(texts) < 2: continue

        input_ids, attention_mask = encode(tokenizer, texts, device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits  # [B,T,V]

        H = res_hook.buffer  # [B,T,D]
        assert H is not None, "Residual hook did not capture."
        side, ratio = token_slice
        mask_eff = _slice_mask(attention_mask, side, ratio)
        RES  = mean_pool_tokens(H, mask_eff).cpu().numpy()             # [B,D]
        UP   = mean_pool_tokens(accessor.UP(H),       mask_eff).cpu().numpy()
        GPRE = mean_pool_tokens(accessor.GATE_PRE(H), mask_eff).cpu().numpy()
        GACT = mean_pool_tokens(accessor.GATE_ACT(H), mask_eff).cpu().numpy()
        POST = mean_pool_tokens(accessor.POST(H),     mask_eff).cpu().numpy()
        DOWN = mean_pool_tokens(accessor.DOWN(accessor.POST(H)), mask_eff).cpu().numpy()

        # Next-token logits at the final position (per text)
        last_pos = attention_mask.sum(dim=1)-1  # [B]
        idxs = last_pos.view(-1,1,1).expand(-1,1,logits.size(-1))
        next_logits = logits.gather(1, idxs).squeeze(1).detach().cpu().numpy()  # [B,V]

        row = dict(prompt_index=idx, prompt_count=ps.prompt_count, N=len(texts),
                   RES=RES, UP=UP, GATE_PRE=GPRE, GATE_ACT=GACT, POST=POST, DOWN=DOWN,
                   NEXTLOGITS=next_logits)
        results["per_prompt"].append(row)

        if (idx+1) % 20 == 0:
            logging.info("[stations CAP] %d/%d — %s", idx+1, len(items), format_eta(idx+1, len(items), start))

    res_hook.close()
    return results

# Dispersion metrics

def _to_mat(arr_list: np.ndarray) -> np.ndarray:
    """Ensure array of shape [B,D]."""
    return np.asarray(arr_list)

def cos_to_first(M: np.ndarray) -> np.ndarray:
    x0 = M[0]; X = M[1:]
    x0n = x0 / (np.linalg.norm(x0) + 1e-12)
    Xn  = X  / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return (Xn @ x0n)  # [B-1]

# L2 helpers
def l2_to_first(M: np.ndarray) -> np.ndarray:
    x0 = M[0]; X = M[1:]
    return np.linalg.norm(X - x0[None, :], axis=1)

def mean_pairwise_one_minus_cos(M: np.ndarray) -> float:
    n = M.shape[0]
    s = 0.0; c = 0
    for i in range(n):
        for j in range(i+1, n):
            s += (1.0 - cosine(M[i], M[j])); c += 1
    return float(s / max(c, 1))

def mean_pairwise_L2(M: np.ndarray) -> float:
    n = M.shape[0]
    s = 0.0; c = 0
    for i in range(n):
        for j in range(i+1, n):
            s += float(np.linalg.norm(M[i] - M[j])); c += 1
    return float(s / max(c, 1))

def centroid_dispersion(M: np.ndarray) -> float:
    C = M.mean(axis=0)
    d = np.array([1.0 - cosine(x, C) for x in M])
    return float(d.mean())

def centroid_dispersion_L2(M: np.ndarray) -> float:
    C = M.mean(axis=0)
    d = np.linalg.norm(M - C[None, :], axis=1)
    return float(d.mean())

def pca_topk_energy(M: np.ndarray, k: int = 8) -> float:
    X = M - M.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return float(S[:k].sum())

def per_dim_var_mean(M: np.ndarray) -> float:
    return float(np.var(M, axis=0).mean())

def dispersion_block(per_prompt_cap: List[Dict[str, Any]], pca_k: int = 8) -> List[Dict[str, Any]]:
    """Compute all dispersion metrics per station for a single model capture list."""
    rows = []
    for r in per_prompt_cap:
        row = {"prompt_count": r["prompt_count"], "N": r["N"]}
        for st in ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]:
            M = _to_mat(r[st])
            # Cosine-based: "distance to original": 1 - mean cosine(orig, paraphrase)
            dist = float(1.0 - cos_to_first(M).mean())
            # L2-based equivalents
            dist_L2 = float(l2_to_first(M).mean())
            row[f"dist_{st}"] = dist
            row[f"dist_L2_{st}"] = dist_L2

            row[f"disp_pairwise_{st}"]   = mean_pairwise_one_minus_cos(M)
            row[f"disp_pairwise_L2_{st}"] = mean_pairwise_L2(M)

            row[f"disp_centroid_{st}"]   = centroid_dispersion(M)
            row[f"disp_centroid_L2_{st}"] = centroid_dispersion_L2(M)

            row[f"disp_pcaK_{st}"]      = pca_topk_energy(M, k=pca_k)
            row[f"disp_varmean_{st}"]   = per_dim_var_mean(M)
        # Stage deltas (cosine-based; retained for csv parity)
        row["delta_dist_POST_minus_UP"]  = row["dist_POST"]  - row["dist_UP"]
        row["delta_dist_DOWN_minus_UP"]  = row["dist_DOWN"]  - row["dist_UP"]
        rows.append(row)
    return rows

def write_csv(rows: List[Dict[str, Any]], path: Path):
    if not rows: return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

def summarize_numeric(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows: return []
    keys = [k for k in rows[0].keys() if isinstance(rows[0][k], (int,float))]
    agg = {k: [] for k in keys}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int,float)) and np.isfinite(v):
                agg.setdefault(k, []).append(float(v))
    out = []
    for stat in ["mean","median","std","min","max","n"]:
        row = {"stat": stat}
        for k, vals in agg.items():
            if not vals: row[k] = float("nan"); continue
            v = np.array(vals, dtype=float)
            if stat == "mean":   row[k] = float(v.mean())
            elif stat == "median": row[k] = float(np.median(v))
            elif stat == "std":  row[k] = float(v.std(ddof=1) if v.size>1 else 0.0)
            elif stat == "min":  row[k] = float(v.min())
            elif stat == "max":  row[k] = float(v.max())
            elif stat == "n":    row[k] = int(v.size)
        out.append(row)
    return out

# Plot helpers

BASE_GRAY = "#6e6e6e"
FT_GREEN  = "#0b5d1e"

def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#"); return tuple(int(hex_color[i:i+2], 16) for i in (0,2,4))

def _rgb_to_hex(rgb): return "#{:02x}{:02x}{:02x}".format(*rgb)

def _lighten(hex_color: str, factor: float):
    r,g,b = _hex_to_rgb(hex_color)
    r = int(r + (255-r)*factor); g = int(g + (255-g)*factor); b = int(b + (255-b)*factor)
    return _rgb_to_hex((r,g,b))

def bar(series: List[float], labels: List[str], title: str, out_path: Path):
    plt.figure(figsize=(10,4.2))
    plt.bar(labels, series, color=_lighten(FT_GREEN, 0.2), edgecolor="none")
    plt.axhline(0.0, color=BASE_GRAY, linewidth=0.9)
    plt.ylabel("ΔΔ (FT−BASE)")
    plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def bar_dual(base_vals: List[float], ft_vals: List[float], labels: List[str], title: str, out_path: Path):
    x = np.arange(len(labels)); w = 0.36
    plt.figure(figsize=(10,4.2))
    plt.bar(x-w/2, base_vals, width=w, color=_lighten(BASE_GRAY,0.25), edgecolor="black", label="BASE")
    plt.bar(x+w/2, ft_vals,   width=w, color=_lighten(FT_GREEN,0.20), edgecolor="none",  label="FT")
    plt.xticks(x, labels, rotation=0); plt.legend()
    plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def bar_abs(values: Dict[str,float], title: str, out_path: Path):
    """Single-figure absolute bars with gray/green palette."""
    labels = list(values.keys()); vals = list(values.values())
    colors = []
    for lab in labels:
        if lab.upper()=="BASE": colors.append(_lighten(BASE_GRAY,0.25))
        elif lab.upper()=="FT": colors.append(_lighten(FT_GREEN,0.20))
        else: colors.append(_lighten(FT_GREEN,0.35))
    plt.figure(figsize=(11,4.0))
    plt.bar(labels, vals, color=colors, edgecolor="none")
    plt.axhline(0.0, color=BASE_GRAY, linewidth=0.9)
    plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

# KL invariance + patching

class MlpPatchContext:
    """Context manager: patch FT MLP at a layer by mixing BASE/FT branches.
       All computations use the *same* hidden H (taken from the FT run) so
       branch comparisons are apples-to-apples.
    """
    def __init__(self, base_model, ft_model, layer: int, mode: str):
        self.base = base_model; self.ft = ft_model; self.layer = layer; self.mode = mode
        self.ft_mlp = self._get_mlp(self.ft, layer)
        self.base_accessor = MLPAccessor(base_model, layer, gate_mode="silu")
        self.ft_accessor   = MLPAccessor(ft_model,   layer, gate_mode="silu")
        self._pre = None; self._hook = None; self._H = None

    def _get_mlp(self, model, i):
        for stem, leaf in [("model.model.layers","mlp"), ("model.layers","mlp"), ("transformer.h","mlp")]:
            try:
                base = model
                for part in stem.split("."): base = getattr(base, part)
                return getattr(base[i], leaf)
            except Exception: pass
        raise RuntimeError("Could not find mlp for patching")

    def __enter__(self):
        def pre_hook(module, inputs):
            self._H = inputs[0].detach()

        def fwd_hook(module, inputs, output):
            H = self._H
            # Build branches
            up_b  = self.base_accessor.UP(H);    up_f  = self.ft_accessor.UP(H)
            gact_b= self.base_accessor.GATE_ACT(H); gact_f= self.ft_accessor.GATE_ACT(H)
            # POSTs
            if self.base_accessor.kind=="swiglu":
                post_b = up_b * gact_b
                post_f = up_f * gact_f
            else:
                post_b = self.base_accessor.POST(H)
                post_f = self.ft_accessor.POST(H)
            # DOWNs
            down_b= self.base_accessor.DOWN(post_b)
            down_f= self.ft_accessor.DOWN(post_f)

            mode = (self.mode or "").lower()
            if mode == "down":
                # Replace whole MLP with BASE
                return down_b
            elif mode == "post":
                # Replace POST only, keep FT down
                return self.ft_accessor.DOWN(post_b)
            elif mode == "post_only":
                # alias
                return self.ft_accessor.DOWN(post_b)
            elif mode == "down_only":
                # Replace DOWN only: use FT POST but BASE down
                return self.base_accessor.DOWN(post_f)
            elif mode == "up":
                # Replace UP only (keep FT gate)
                if self.ft_accessor.kind=="swiglu":
                    post = up_b * gact_f
                    return self.ft_accessor.DOWN(post)
                else:
                    # GeLU: replacing UP indistinguishable from replacing POST
                    return self.ft_accessor.DOWN(self.base_accessor.POST(H))
            elif mode == "gate":
                # Replace GATE only (keep FT up)
                if self.ft_accessor.kind=="swiglu":
                    post = up_f * gact_b
                    return self.ft_accessor.DOWN(post)
                else:
                    return self.ft_accessor.DOWN(self.base_accessor.POST(H))
            elif mode == "up+down":
                # Replace UP and DOWN (keep FT gate)
                if self.ft_accessor.kind=="swiglu":
                    post = up_b * gact_f
                    return self.base_accessor.DOWN(post)
                else:
                    return down_b
            elif mode == "post+down":
                # Replace POST and DOWN → equals full DOWN in SwiGLU
                return down_b
            else:
                return output  # no-op

        self._pre  = self.ft_mlp.register_forward_pre_hook(pre_hook)
        self._hook = self.ft_mlp.register_forward_hook(fwd_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._pre is not None: self._pre.remove()
            if self._hook is not None: self._hook.remove()
        except Exception: pass
        return False

def logits_for_texts(model, tokenizer, texts: List[str], device: torch.device, batch: int = 8) -> torch.Tensor:
    """Return next-token logits [B,V] for a list of texts."""
    outs = []
    for chunk in batched(texts, batch):
        input_ids, attention_mask = encode(tokenizer, chunk, device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
        last = attention_mask.sum(dim=1)-1
        idxs = last.view(-1,1,1).expand(-1,1,logits.size(-1))
        outs.append(logits.gather(1, idxs).squeeze(1))
    return torch.cat(outs, dim=0)

def kl_invariance_grid(base, tok_b, ft, tok_f, items: List[PromptSet], device: torch.device,
                       layer: int, patch_modes: List[str], batch: int = 8,
                       max_prompts: Optional[int] = None, max_paraphrases: Optional[int] = None) -> Dict[str, Any]:
    """Compute ΔΔ symmetric-KL and patching effects (per mode)."""
    out = {"rows": []}
    start = time.time()
    logging.info("[KL grid] BEGIN")
    for i, ps in enumerate(items):
        if max_prompts is not None and i >= max_prompts: break
        texts = [ps.instruction_original] + [t for _, t in ps.paraphrases]
        if max_paraphrases is not None: texts = texts[:1+max_paraphrases]
        if len(texts) < 2: continue

        # BASE and FT
        logits_b = logits_for_texts(base, tok_b, texts, device, batch=batch)
        logits_f = logits_for_texts(ft,   tok_f, texts, device, batch=batch)

        # symmetric KL( orig ↔ paraphrase ), averaged across paraphrases
        p0_b = logits_b[0:1].expand(logits_b.size(0)-1, -1)
        p0_f = logits_f[0:1].expand(logits_f.size(0)-1, -1)
        kl_b = symmetric_kl(p0_b, logits_b[1:]).mean().item()
        kl_f = symmetric_kl(p0_f, logits_f[1:]).mean().item()

        row = {"prompt_count": ps.prompt_count, "kl_BASE": kl_b, "kl_FT": kl_f}
        # Patching runs on FT
        for mode in patch_modes:
            with MlpPatchContext(base, ft, layer, mode):
                logits_p = logits_for_texts(ft, tok_f, texts, device, batch=batch)
            p0_p = logits_p[0:1].expand(logits_p.size(0)-1, -1)
            kl_p = symmetric_kl(p0_p, logits_p[1:]).mean().item()
            row[f"kl_patch_{mode}"] = kl_p
        out["rows"].append(row)

        if (i+1) % 20 == 0:
            logging.info("[KL grid] %d/%d — %s", i+1, len(items), format_eta(i+1, len(items), start))
    logging.info("[KL grid] DONE (%d rows)", len(out["rows"]))
    return out

#== Logit-sensitive subspace==

def _ridge_regression(X: torch.Tensor, Y: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    """Solve min_W ||XW - Y||^2 + lam ||W||^2 . Returns W [D,V]."""
    D = X.shape[1]; I = torch.eye(D, device=X.device, dtype=X.dtype)
    A = X.T @ X + lam * I
    W = torch.linalg.solve(A, X.T @ Y)
    return W  # [D,V]

def build_sensitive_subspace(per_prompt_cap: List[Dict[str, Any]], vocab_dim: int,
                             topk_vocab: int = 128, lam: float = 1e-3, topk_sv: int = 8) -> List[Dict[str, Any]]:
    """
    For each station, for each prompt-set:
      X := ΔH (paraphrase - original)  [m,D]
      Y := Δlogits (restricted vocab)  [m,Vk]
      W := argmin ||XW - Y||^2 + lam ||W||^2  [D,Vk]
      SVD(W) = U Σ V^T. Project X onto span(U_k) vs orth complement -> report variances.
    """
    rows = []
    stations = ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]
    start = time.time()
    for idx, r in enumerate(per_prompt_cap):
        next_logits = torch.from_numpy(r["NEXTLOGITS"])  # [B,V]
        # Choose a compact vocab subset: union of top tokens over original & paras (up to topk_vocab)
        with torch.no_grad():
            top_idx = torch.topk(next_logits, k=min(topk_vocab, vocab_dim), dim=-1).indices
            idx_set = torch.unique(top_idx.reshape(-1))
            if idx_set.numel() > topk_vocab: idx_set = idx_set[:topk_vocab]
            Vsel = idx_set.to(torch.long)
        row = {"prompt_count": r["prompt_count"]}
        for st in stations:
            M = torch.from_numpy(r[st])  # [B,D]
            X = M[1:] - M[0:1]           # [m,D]
            if X.size(0) < 2:
                sens_var = float("nan"); orth_var = float("nan")
            else:
                Y = next_logits[1:, Vsel] - next_logits[0:1, Vsel]  # [m,Vk]
                Xc = X - X.mean(dim=0, keepdim=True)
                Yc = Y - Y.mean(dim=0, keepdim=True)
                W = _ridge_regression(Xc, Yc, lam=lam)              # [D,Vk]
                # Sensitive subspace = span(U_k)
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                k = min(topk_sv, U.shape[1])
                Uk = U[:, :k]
                P  = Uk @ Uk.T
                X_proj = (Xc @ P)
                X_orth = Xc - X_proj
                sens_var = float(X_proj.pow(2).sum(dim=1).mean().item())
                orth_var = float(X_orth.pow(2).sum(dim=1).mean().item())
            row[f"sens_var_{st}"] = sens_var
            row[f"orth_var_{st}"] = orth_var
        rows.append(row)

        if (idx+1) % 20 == 0:
            logging.info("[sensitive-subspace] %d/%d — %s", idx+1, len(per_prompt_cap),
                         format_eta(idx+1, len(per_prompt_cap), start))
    logging.info("[sensitive-subspace] DONE (%d rows)", len(rows))
    return rows

#=== Main===

def main():
    ap = argparse.ArgumentParser(description="Unified MLP mechanism suite")
    # Preferred flags (accept legacy aliases too)
    ap.add_argument("--selection_jsonl", type=str, required=True)
    ap.add_argument("--base_model_name_or_path", "--base_model", dest="base_model_name_or_path", type=str, required=True)
    ap.add_argument("--ft_model_name_or_path", "--ft_model", dest="ft_model_name_or_path", type=str, default=None)
    ap.add_argument("--ft_lora_adapter", type=str, default=None)
    ap.add_argument("--layer_index", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--max_paraphrases", type=int, default=None)
    ap.add_argument("--token_slice", type=str, default="all", help='e.g. "head:0.7"')
    ap.add_argument("--pca_k", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--min_content_score", type=int, default=None)
    ap.add_argument("--allow_instruct_types", type=str, default=None,
                    help="comma-separated keys/types to keep; requires they exist in JSONL")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
        datefmt=":%H:%M:%S",
    )
    set_seed(args.seed)
    device = torch.device(args.device)
    outdir = ensure_dir(args.outdir)

    allow_types = None
    if args.allow_instruct_types:
        allow_types = set(t.strip() for t in args.allow_instruct_types.split(",") if t.strip())

    logging.info("==== STAGE 1/6: Reading selection ====")
    logging.info("Reading selection: %s", args.selection_jsonl)
    items = load_selection_jsonl(args.selection_jsonl, max_prompts=args.max_prompts,
                                 min_content_score=args.min_content_score,
                                 allow_instruct_types=allow_types)
    logging.info("Loaded %d prompt groups.", len(items))

    logging.info("==== STAGE 2/6: Loading models ====")
    logging.info("Loading models: BASE=%s | FT=%s | LoRA=%s",
                 args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter)
    (base, tok_b), (ft, tok_f) = build_models_and_tokenizers(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, device
    )

    logging.info("==== STAGE 3/6: Station capture (BASE & FT) ====")
    token_slice = _parse_token_slice(args.token_slice)
    logging.info("Capturing stations (BASE) — layer %d", args.layer_index)
    capB = capture_stations(base, tok_b, items, args.layer_index, device, token_slice,
                             args.max_prompts, args.max_paraphrases)
    logging.info("Capturing stations (FT) — layer %d", args.layer_index)
    capF = capture_stations(ft, tok_f, items, args.layer_index, device, token_slice,
                             args.max_prompts, args.max_paraphrases)

    # cache
    np.save(outdir / "cap_BASE.npy", capB, allow_pickle=True)
    np.save(outdir / "cap_FT.npy",   capF, allow_pickle=True)

    logging.info("==== STAGE 4/6: Dispersion metrics + ΔΔ ====")
    dispB = dispersion_block(capB["per_prompt"], pca_k=args.pca_k)
    dispF = dispersion_block(capF["per_prompt"], pca_k=args.pca_k)

    # Merge ΔΔ
    bm = {r["prompt_count"]: r for r in dispB}
    fm = {r["prompt_count"]: r for r in dispF}
    merged = []
    for pc in sorted(set(bm.keys()) & set(fm.keys())):
        rb, rf = bm[pc], fm[pc]
        row = {"prompt_count": pc, "N": rb["N"]}
        for st in ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]:
            # include both cosine-based and L2-based keys
            for key in ["dist","dist_L2","disp_pairwise","disp_pairwise_L2","disp_centroid","disp_centroid_L2","disp_pcaK","disp_varmean"]:
                row[f"{key}_{st}_BASE"] = rb[f"{key}_{st}"]
                row[f"{key}_{st}_FT"]   = rf[f"{key}_{st}"]
                row[f"ddiff_{key}_{st}"] = rf[f"{key}_{st}"] - rb[f"{key}_{st}"]
        for k in ["delta_dist_POST_minus_UP","delta_dist_DOWN_minus_UP"]:
            row[k+"_BASE"] = rb[k]; row[k+"_FT"] = rf[k]
            row["ddiff_"+k] = rf[k] - rb[k]
        merged.append(row)

    write_csv(merged, outdir / f"merged_dispersion_layer{args.layer_index}.csv")
    write_csv(summarize_numeric(merged), outdir / f"summaries_dispersion_layer{args.layer_index}.csv")

    # Plots: use L2 variants for distance graphics
    steps = ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]

    def ddiff_means(metric: str) -> List[float]:
        vals = []
        for s in steps:
            arr = np.array([r[f"ddiff_{metric}_{s}"] for r in merged], dtype=float)
            vals.append(float(arr.mean()) if arr.size else float("nan"))
        return vals

    # L2-based figures
    bar(ddiff_means("disp_centroid_L2"), steps, "ΔΔ L2 distance to centroid across MLP steps",
        outdir / "ddiff_disp_centroid_bars.png")
    bar(ddiff_means("disp_pairwise_L2"), steps, "ΔΔ Mean pairwise L2 across MLP steps",
        outdir / "ddiff_disp_pairwise_bars.png")
    # keep PCA/varmean as-is
    bar(ddiff_means("disp_pcaK"), steps, "ΔΔ PCA top-k energy (Σσ) across MLP steps",
        outdir / "ddiff_disp_pcaK_bars.png")
    bar(ddiff_means("disp_varmean"), steps, "ΔΔ Mean per-dim variance across MLP steps",
        outdir / "ddiff_disp_varmean_bars.png")
    bar(ddiff_means("dist_L2"), steps, "ΔΔ L2 distance to original across MLP steps",
        outdir / "ddiff_dist_bars.png")

    logging.info("==== STAGE 5/6: KL invariance + patching grid ====")
    patch_modes = ["DOWN", "DOWN_ONLY", "POST", "POST_ONLY", "UP", "GATE", "UP+DOWN", "POST+DOWN"]
    grid = kl_invariance_grid(base, tok_b, ft, tok_f, items, device, args.layer_index,
                              patch_modes=patch_modes, batch=args.batch_size,
                              max_prompts=args.max_prompts, max_paraphrases=args.max_paraphrases)
    rows = grid["rows"]
    write_csv(rows, outdir / f"kl_patch_rows_layer{args.layer_index}.csv")

    # Summaries
    def smean(name: str) -> float:
        v = [r[name] for r in rows if (name in r and np.isfinite(r[name]))]; 
        return float(np.mean(v)) if v else float("nan")

    kl_base = smean("kl_BASE")
    kl_ft   = smean("kl_FT")
    kl_dd   = kl_ft - kl_base

    summ = [{"stat":"mean", "kl_BASE":kl_base, "kl_FT":kl_ft, "kl_ddiff":kl_dd}]
    for m in patch_modes:
        kp = smean(f"kl_patch_{m}")
        summ[0][f"kl_patch_{m}"] = kp
        summ[0][f"delta_patch_{m}"] = kp - kl_ft
    write_csv(summ, outdir / f"kl_patch_summaries_layer{args.layer_index}.csv")

    # Plots: (1) ΔΔ KL; (2) absolute KL including patched; (3) delta vs F
    plt.figure(figsize=(4.8,3.6))
    plt.bar(["KL ΔΔ"], [kl_dd], color=_lighten(FT_GREEN,0.20), edgecolor="none")
    plt.axhline(0, color=BASE_GRAY, linewidth=0.9)
    plt.ylabel("ΔΔ symmetric KL (orig↔para)")
    plt.title("Next-token invariance")
    plt.tight_layout(); plt.savefig(outdir / "logits_kl_ddiff_bar.png", dpi=160); plt.close()

    # Absolute
    abs_vals = {"BASE": kl_base, "FT": kl_ft}
    for m in patch_modes: abs_vals[m] = summ[0][f"kl_patch_{m}"]
    bar_abs(abs_vals, "Absolute symmetric-KL (lower = more invariant)", outdir / "logits_kl_absolute_bars.png")

    # Delta vs FT
    delta_vals = {m: summ[0][f"delta_patch_{m}"] for m in patch_modes}
    bar_abs(delta_vals, "Change in symmetric-KL vs FT after patch", outdir / "logits_kl_delta_vsFT_bars.png")

    logging.info("==== STAGE 6/6: Logit-sensitive subspace ====")
    sensB = build_sensitive_subspace(capB["per_prompt"], vocab_dim=tok_b.vocab_size, topk_vocab=128, lam=1e-3, topk_sv=8)
    sensF = build_sensitive_subspace(capF["per_prompt"], vocab_dim=tok_f.vocab_size, topk_vocab=128, lam=1e-3, topk_sv=8)
    write_csv(sensB, outdir / f"sens_subspace_BASE_layer{args.layer_index}.csv")
    write_csv(summarize_numeric(sensB), outdir / f"sens_subspace_BASE_summary_layer{args.layer_index}.csv")
    write_csv(sensF, outdir / f"sens_subspace_FT_layer{args.layer_index}.csv")
    write_csv(summarize_numeric(sensF), outdir / f"sens_subspace_FT_summary_layer{args.layer_index}.csv")

    # Compare BASE vs FT (means across prompts) — sensitive vs orth variance
    def sens_means(rows: List[Dict[str, Any]], tag: str) -> List[float]:
        out = []
        for st in ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]:
            key = f"{tag}_{st}"
            vals = [r[key] for r in rows if (key in r and np.isfinite(r[key]))]
            out.append(float(np.mean(vals)) if vals else float("nan"))
        return out

    steps2 = ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]
    base_sens = sens_means(sensB, "sens_var"); base_orth = sens_means(sensB, "orth_var")
    ft_sens   = sens_means(sensF, "sens_var"); ft_orth   = sens_means(sensF, "orth_var")

    bar_dual(base_sens, ft_sens, steps2, "Sensitive var (ΔH → Δlogits subspace)", outdir / "sens_subspace_sens_var_bars.png")
    bar_dual(base_orth, ft_orth, steps2, "Orthogonal var (ΔH ⟂ subspace)", outdir / "sens_subspace_orth_var_bars.png")

    logging.info("==== ALL DONE ====")
    logging.info("Outputs in: %s", outdir)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
