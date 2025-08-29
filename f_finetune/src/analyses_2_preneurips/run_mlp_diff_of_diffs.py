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

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

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
    items = []
    n = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            paraphrases = [(p["key"], p["text"]) for p in obj.get("paraphrases", []) if isinstance(p, dict)]
            items.append(PromptSet(
                prompt_count=int(obj["prompt_count"]),
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

def mean_pool_tokens_for_slices(x: torch.Tensor, mask: torch.Tensor, n_slices: int) -> List[Optional[np.ndarray]]:
    """
    Mean-pool x over contiguous slices along valid (mask==1) token positions.
    Returns a list of length n_slices with np.ndarray vectors (or None if slice is empty).
    x: [B=1, T, D]; mask: [B=1, T].
    """
    assert x.dim() == 3 and mask.dim() == 2 and x.size(0) == 1 and mask.size(0) == 1
    with torch.no_grad():
        pos = (mask[0] > 0).nonzero(as_tuple=False).squeeze(1).cpu().numpy()
        n = int(pos.size)
        if n == 0 or n_slices <= 0:
            return [None] * max(n_slices, 0)
        cuts = np.linspace(0, n, n_slices + 1).astype(int)
        out: List[Optional[np.ndarray]] = []
        for i in range(n_slices):
            sel = pos[cuts[i]:cuts[i + 1]]
            if sel.size == 0:
                out.append(None)
            else:
                m = torch.zeros_like(mask[0], dtype=x.dtype)
                m[torch.from_numpy(sel).to(m.device)] = 1
                s = (x[0] * m.unsqueeze(-1)).sum(dim=0)  # [D]
                c = m.sum().clamp(min=1.0)
                out.append( (s / c).to(torch.float32).cpu().numpy() )
        return out

class MLPAccessor:
    """
    Accessor that (1) finds the model's MLP module at a layer, and
    (2) uses the MLP's *actual* nonlinearity as defined by the module.

    For SwiGLU MLPs (e.g., Gemma/LLaMA families), we honor `mlp.act_fn` if present;
    fallback is SiLU. For GeLU-style MLPs, POST = gelu(pre), as in standard HF modules.
    """
    def __init__(self, model, layer: int):
        self.model = model
        self.layer = layer
        self.mlp = self._get_mlp_module(model, layer)
        self.kind = self._detect_kind(self.mlp)
        p = next(self.mlp.parameters())
        self.device = p.device
        self.dtype = p.dtype

        # Try to get the module's own activation function
        # Common HF patterns: act_fn (callable), activation_fn (callable)
        self._act_fn = getattr(self.mlp, "act_fn", None)
        if self._act_fn is None:
            self._act_fn = getattr(self.mlp, "activation_fn", None)

        # Robust fallback:
        if self._act_fn is None:
            import torch.nn.functional as F
            if self.kind == "swiglu":
                self._act_fn = F.silu
            else:
                self._act_fn = F.gelu

    def _get_mlp_module(self, model, i: int):
        for stem, leaf in [("model.model.layers","mlp"), ("model.layers","mlp"), ("transformer.h","mlp")]:
            try:
                base = model
                for part in stem.split("."):
                    base = getattr(base, part)
                return getattr(base[i], leaf)
            except Exception:
                pass
        raise RuntimeError("Could not locate MLP for layer {}".format(i))

    def _detect_kind(self, mlp) -> str:
        if hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
            return "swiglu"
        if any(hasattr(mlp, n) for n in ["wi","fc_in","dense_h_to_4h"]):
            return "gelu"
        raise RuntimeError("Unknown MLP kind (not SwiGLU/GeLU)")

    @torch.no_grad()
    def UP(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu":
            return self.mlp.up_proj(h)
        else:
            if hasattr(self.mlp, "wi"): return self.mlp.wi(h)
            if hasattr(self.mlp, "fc_in"): return self.mlp.fc_in(h)
            if hasattr(self.mlp, "dense_h_to_4h"): return self.mlp.dense_h_to_4h(h)
            raise RuntimeError("Cannot find GeLU upstream linear")

    @torch.no_grad()
    def GATE_PRE(self, h: torch.Tensor) -> torch.Tensor:
        """Linear gate branch (pre-activation). For GeLU MLPs this equals the UP preact."""
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu":
            return self.mlp.gate_proj(h)
        else:
            # GeLU-style MLP has a single pre-activation; mirror UP()
            if hasattr(self.mlp, "wi"): return self.mlp.wi(h)
            if hasattr(self.mlp, "fc_in"): return self.mlp.fc_in(h)
            if hasattr(self.mlp, "dense_h_to_4h"): return self.mlp.dense_h_to_4h(h)
            raise RuntimeError("Cannot find GeLU upstream linear")

    @torch.no_grad()
    def GATE_ACT(self, h: torch.Tensor) -> torch.Tensor:
        """Gate branch after nonlinearity, before multiplication with UP."""
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu":
            gate_lin = self.mlp.gate_proj(h)
            return self._act_fn(gate_lin)
        else:
            # GeLU MLP: activation is GeLU(pre), which is also the POST input.
            if   hasattr(self.mlp, "wi"):            pre = self.mlp.wi(h)
            elif hasattr(self.mlp, "fc_in"):         pre = self.mlp.fc_in(h)
            elif hasattr(self.mlp, "dense_h_to_4h"): pre = self.mlp.dense_h_to_4h(h)
            else:
                raise RuntimeError("Cannot find GeLU upstream linear")
            import torch.nn.functional as F
            return F.gelu(pre)

    @torch.no_grad()
    def POST(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.kind == "swiglu":
            up = self.mlp.up_proj(h)
            gate_lin = self.mlp.gate_proj(h)
            gate = self._act_fn(gate_lin)
            return up * gate
        else:
            if hasattr(self.mlp, "wi"):
                import torch.nn.functional as F
                return F.gelu(self.mlp.wi(h))
            if hasattr(self.mlp, "fc_in"):
                import torch.nn.functional as F
                return F.gelu(self.mlp.fc_in(h))
            if hasattr(self.mlp, "dense_h_to_4h"):
                import torch.nn.functional as F
                return F.gelu(self.mlp.dense_h_to_4h(h))
            raise RuntimeError("Cannot find GeLU downstream POST")

    @torch.no_grad()
    def DOWN(self, post: torch.Tensor) -> torch.Tensor:
        post = post.to(self.device, self.dtype)
        if hasattr(self.mlp, "down_proj"): return self.mlp.down_proj(post)
        if hasattr(self.mlp, "wo"): return self.mlp.wo(post)
        if hasattr(self.mlp, "fc_out"): return self.mlp.fc_out(post)
        if hasattr(self.mlp, "dense_4h_to_h"): return self.mlp.dense_4h_to_h(post)
        raise RuntimeError("Cannot find GeLU downstream linear")

class ResidualHook:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.buffer = None
        self.mlp = self._get_mlp(model, layer)
        self.hook = self.mlp.register_forward_pre_hook(self._hook)

    def _get_mlp(self, model, i):
        for stem, leaf in [("model.model.layers","mlp"), ("model.layers","mlp"), ("transformer.h","mlp")]:
            try:
                base = model
                for part in stem.split("."):
                    base = getattr(base, part)
                return getattr(base[i], leaf)
            except Exception:
                pass
        raise RuntimeError("Could not find mlp for residual hook")

    def _hook(self, module, inputs):
        self.buffer = inputs[0].detach()  # [B,T,D]

    def close(self):
        try: self.hook.remove()
        except Exception: pass

def cosine_to_original(mat: np.ndarray) -> float:
    if mat.shape[0] < 2:
        return float("nan")
    x0 = mat[0]
    X  = mat[1:]
    x0n = x0 / (np.linalg.norm(x0) + 1e-12)
    Xn  = X  / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos = (Xn @ x0n)
    return float(cos.mean())

def avg_l2_to_original(mat: np.ndarray) -> float:
    if mat.shape[0] < 2:
        return float("nan")
    x0 = mat[0]
    diffs = mat[1:] - x0[None, :]
    d = np.sqrt((diffs**2).sum(axis=1))
    return float(d.mean())

@torch.no_grad()
def capture_stations_for_model(model, tokenizer, items: List[PromptSet], layer: int, device: torch.device,
                               max_prompts: Optional[int], max_paraphrases: Optional[int],
                               token_slices: int = 0) -> Dict[str, Any]:
    results = {"layer": layer, "per_prompt": [], "n_used": 0}
    accessor = MLPAccessor(model, layer)
    res_hook = ResidualHook(model, layer)

    start = time.time()
    for idx, ps in enumerate(items):
        texts = [ps.instruction_original] + [t for _, t in ps.paraphrases]
        if max_paraphrases is not None:
            texts = texts[: 1 + max_paraphrases]
        if len(texts) < 2:
            logging.warning("Prompt %s has <2 texts; skipping.", ps.prompt_count)
            continue

        S = int(token_slices) if token_slices and token_slices > 0 else 0
        if S > 0:
            # For each stage, keep a list per-slice, each holding per-text vectors (None if unavailable)
            per_slice = {
                "RES":       [[None] * len(texts) for _ in range(S)],
                "UP":        [[None] * len(texts) for _ in range(S)],
                "GATE_PRE":  [[None] * len(texts) for _ in range(S)],
                "GATE_ACT":  [[None] * len(texts) for _ in range(S)],
                "POST":      [[None] * len(texts) for _ in range(S)],
                "DOWN":      [[None] * len(texts) for _ in range(S)],
            }

        UP_rows: List[np.ndarray] = []
        POST_rows: List[np.ndarray] = []

        GATEPRE_rows: List[np.ndarray] = []
        GATEACT_rows: List[np.ndarray] = []

        DOWN_rows: List[np.ndarray] = []
        RES_rows: List[np.ndarray] = []

        for j, t in enumerate(texts):
            input_ids, attention_mask = encode(tokenizer, t, device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

            H = res_hook.buffer
            res_vec = mean_pool_tokens(H, attention_mask).cpu().numpy()
            RES_rows.append(res_vec)

            up = accessor.UP(H)
            gate_pre = accessor.GATE_PRE(H)
            gate_act = accessor.GATE_ACT(H)
            post = accessor.POST(H)   # equals up * gate_act (SwiGLU) or GeLU(pre) (GeLU)
            down = accessor.DOWN(post)

            if S > 0:
                # Per-slice pooling per stage for this text index j
                res_s   = mean_pool_tokens_for_slices(H,        attention_mask, S)
                up_s    = mean_pool_tokens_for_slices(up,       attention_mask, S)
                gpre_s  = mean_pool_tokens_for_slices(gate_pre, attention_mask, S)
                gact_s  = mean_pool_tokens_for_slices(gate_act, attention_mask, S)
                post_s  = mean_pool_tokens_for_slices(post,     attention_mask, S)
                down_s  = mean_pool_tokens_for_slices(down,     attention_mask, S)
                for si in range(S):
                    if res_s[si]  is not None: per_slice["RES"][si][j]      = res_s[si]
                    if up_s[si]   is not None: per_slice["UP"][si][j]       = up_s[si]
                    if gpre_s[si] is not None: per_slice["GATE_PRE"][si][j] = gpre_s[si]
                    if gact_s[si] is not None: per_slice["GATE_ACT"][si][j] = gact_s[si]
                    if post_s[si] is not None: per_slice["POST"][si][j]     = post_s[si]
                    if down_s[si] is not None: per_slice["DOWN"][si][j]     = down_s[si]

            up_vec   = mean_pool_tokens(up, attention_mask).cpu().numpy()
            post_vec = mean_pool_tokens(post, attention_mask).cpu().numpy()
            gatepre_vec = mean_pool_tokens(gate_pre, attention_mask).cpu().numpy()
            gateact_vec = mean_pool_tokens(gate_act, attention_mask).cpu().numpy()

            down_vec = mean_pool_tokens(down, attention_mask).cpu().numpy()

            GATEPRE_rows.append(gatepre_vec); GATEACT_rows.append(gateact_vec)
            UP_rows.append(up_vec); POST_rows.append(post_vec); DOWN_rows.append(down_vec)

        RES  = np.stack(RES_rows,  axis=0)
        UP   = np.stack(UP_rows,   axis=0)

        POST = np.stack(POST_rows, axis=0)

        GATE_PRE = np.stack(GATEPRE_rows, axis=0)
        GATE_ACT = np.stack(GATEACT_rows, axis=0)

        DOWN = np.stack(DOWN_rows, axis=0)

        # Ensure we have L2 for all six stages, including RES:
        l2_RES  = avg_l2_to_original(RES)
        l2_UP   = avg_l2_to_original(UP)
        l2_POST = avg_l2_to_original(POST)
        l2_DOWN = avg_l2_to_original(DOWN)
        l2_GATE_PRE = avg_l2_to_original(GATE_PRE)
        l2_GATE_ACT = avg_l2_to_original(GATE_ACT)

        row = {
            "prompt_index": idx,
            "prompt_count": ps.prompt_count,
            "N": int(RES.shape[0]),
            "cos_RES": cosine_to_original(RES),
            "cos_UP": cosine_to_original(UP),
            "cos_POST": cosine_to_original(POST),
            "cos_DOWN": cosine_to_original(DOWN),
            "dist_RES": float(1.0 - cosine_to_original(RES)),
            "dist_UP": float(1.0 - cosine_to_original(UP)),
            "dist_POST": float(1.0 - cosine_to_original(POST)),
            "dist_DOWN": float(1.0 - cosine_to_original(DOWN)),
            "l2_RES": l2_RES,
            "l2_UP": l2_UP,
            "l2_POST": l2_POST,
            "l2_DOWN": l2_DOWN,
            "delta_dist_POST_minus_UP": float((1.0 - cosine_to_original(POST)) - (1.0 - cosine_to_original(UP))),
            "delta_dist_DOWN_minus_UP": float((1.0 - cosine_to_original(DOWN)) - (1.0 - cosine_to_original(UP))),
        }
        row.update({
            "cos_GATE_PRE": cosine_to_original(GATE_PRE),
            "cos_GATE_ACT": cosine_to_original(GATE_ACT),
            "dist_GATE_PRE": float(1.0 - cosine_to_original(GATE_PRE)),
            "dist_GATE_ACT": float(1.0 - cosine_to_original(GATE_ACT)),
            "l2_GATE_PRE": l2_GATE_PRE,
            "l2_GATE_ACT": l2_GATE_ACT,
        })

        # Per-token-slice metrics (six stages)
        if S > 0:
            stages6 = ["RES", "UP", "GATE_PRE", "GATE_ACT", "POST", "DOWN"]
            for si in range(S):
                # We only compute slice metrics if the ORIGINAL (text index 0) has tokens in this slice.
                for st in stages6:
                    vecs = per_slice[st][si]  # list length = len(texts), entries np.ndarray or None
                    if vecs[0] is None:
                        # No baseline for this slice; store NaNs
                        row[f"cos_{st}_S{si+1}"]  = float("nan")
                        row[f"dist_{st}_S{si+1}"] = float("nan")
                        row[f"l2_{st}_S{si+1}"]   = float("nan")
                        continue
                    # Build matrix: original first, then any paraphrase that has this slice
                    mat_list = [vecs[0]] + [v for v in vecs[1:] if v is not None]
                    if len(mat_list) < 2:
                        row[f"cos_{st}_S{si+1}"]  = float("nan")
                        row[f"dist_{st}_S{si+1}"] = float("nan")
                        row[f"l2_{st}_S{si+1}"]   = float("nan")
                        continue
                    MAT = np.stack(mat_list, axis=0)
                    c = cosine_to_original(MAT)
                    row[f"cos_{st}_S{si+1}"]  = c
                    row[f"dist_{st}_S{si+1}"] = float(1.0 - c)
                    row[f"l2_{st}_S{si+1}"]   = avg_l2_to_original(MAT)

        results["per_prompt"].append(row)
        results["n_used"] += 1

        if (idx+1) % 20 == 0:
            logging.info("[stations] %d/%d prompts processed — %s", idx+1, len(items), format_eta(idx+1, len(items), start))

    res_hook.close()
    return results

def normalize_t(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v = v.to(torch.float32)
    n = v.norm(p=2)
    return v / (n + eps)

def unit_random_like(h: torch.Tensor) -> torch.Tensor:
    r = torch.randn_like(h, dtype=torch.float32)
    return normalize_t(r)

@torch.no_grad()
def jacobian_norms_for_model(model, tokenizer, items: List[PromptSet], layer: int,
                             device: torch.device, topk_pca: int, mode: str,
                             max_prompts: Optional[int], max_paraphrases: Optional[int],
                             eps: float = 1e-3, directions_random: int = 8) -> Dict[str, Any]:
    accessor = MLPAccessor(model, layer)
    res_hook = ResidualHook(model, layer)

    out_rows = []
    start = time.time()
    for idx, ps in enumerate(items):
        texts = [ps.instruction_original] + [t for _, t in ps.paraphrases]
        if max_paraphrases is not None:
            texts = texts[: 1 + max_paraphrases]
        if len(texts) < 3:
            continue

        H_rows = []
        for t in texts:
            input_ids, attention_mask = encode(tokenizer, t, device)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            H = res_hook.buffer
            vec = mean_pool_tokens(H, attention_mask).to(torch.float32)
            H_rows.append(vec)
        H_mat = torch.stack(H_rows, dim=0)
        Xc = H_mat - H_mat.mean(dim=0, keepdim=True)

        mean_dir = normalize_t(H_mat.mean(dim=0))
        dirs = [mean_dir]

        if mode == "pca":
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            k = min(topk_pca, Vh.shape[0])
            for i in range(k):
                dirs.append(normalize_t(Vh[i]))
        else:
            idxs = torch.randperm(Xc.shape[0])[:max(1, topk_pca)]
            for i in idxs:
                dirs.append(normalize_t(Xc[i]))

        for _ in range(directions_random):
            dirs.append(unit_random_like(mean_dir))

        D = torch.stack(dirs, dim=0)
        C = H_mat.mean(dim=0, keepdim=True)

        def f_up(H_):
            return accessor.UP(H_.unsqueeze(0)).squeeze(0)
        def f_down(H_):
            post = accessor.POST(H_.unsqueeze(0)).squeeze(0)
            return accessor.DOWN(post)

        norms_up = []
        norms_down = []
        for j in range(D.shape[0]):
            d = D[j].unsqueeze(0)
            Yp = f_up(C + eps * d); Ym = f_up(C - eps * d)
            G = (Yp - Ym) / (2.0 * eps)
            norms_up.append(float(torch.linalg.vector_norm(G).item()))

            Yp2 = f_down(C + eps * d); Ym2 = f_down(C - eps * d)
            G2 = (Yp2 - Ym2) / (2.0 * eps)
            norms_down.append(float(torch.linalg.vector_norm(G2).item()))

        row = {
            "prompt_index": idx,
            "prompt_count": ps.prompt_count,
            "K": int(D.shape[0]),
            "jac_MEAN_up": norms_up[0],
            "jac_MEAN_down": norms_down[0],
        }
        var_slice = norms_up[1:1+topk_pca]
        rnd_slice = norms_up[1+topk_pca:]
        row["jac_VAR_up_mean"] = float(np.mean(var_slice)) if len(var_slice)>0 else float("nan")
        row["jac_RND_up_mean"] = float(np.mean(rnd_slice)) if len(rnd_slice)>0 else float("nan")

        var_slice_d = norms_down[1:1+topk_pca]
        rnd_slice_d = norms_down[1+topk_pca:]
        row["jac_VAR_down_mean"] = float(np.mean(var_slice_d)) if len(var_slice_d)>0 else float("nan")
        row["jac_RND_down_mean"] = float(np.mean(rnd_slice_d)) if len(rnd_slice_d)>0 else float("nan")

        out_rows.append(row)

        if (idx+1) % 20 == 0:
            logging.info("[jacobian] %d/%d prompts processed — %s", idx+1, len(items), format_eta(idx+1, len(items), start))

    res_hook.close()
    return {"per_prompt": out_rows, "layer": layer}

def ttest_paired(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 2: return float("nan")
    d = a - b
    m = float(np.mean(d)); s = float(np.std(d, ddof=1))
    t = m / (s / max(np.sqrt(len(d)),1.0)) if s > 0 else np.inf
    from math import erf
    p = 2.0 * (1.0 - 0.5*(1.0+erf(abs(t)/math.sqrt(2))))
    return p

def summarize_vec(v: np.ndarray) -> Dict[str, float]:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(mean=float("nan"), median=float("nan"), std=float("nan"), min=float("nan"), max=float("nan"), n=0)
    return dict(mean=float(np.mean(v)), median=float(np.median(v)), std=float(np.std(v, ddof=1) if v.size>1 else 0.0),
                min=float(np.min(v)), max=float(np.max(v)), n=int(v.size))

def mean_ci(vals: np.ndarray) -> Tuple[float,float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0: return 0.0, 0.0
    m = float(vals.mean())
    s = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    ci = 1.96 * s / max(1.0, math.sqrt(vals.size))
    return m, ci

def _summarize_rows_numeric(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    # Collect numeric keys
    nums: Dict[str, List[float]] = {}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                nums.setdefault(k, []).append(float(v))
    stats = {
        "mean":  lambda x: float(np.mean(x)) if x else float("nan"),
        "median":lambda x: float(np.median(x)) if x else float("nan"),
        "std":   lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min":   lambda x: float(np.min(x)) if x else float("nan"),
        "max":   lambda x: float(np.max(x)) if x else float("nan"),
    }
    out = []
    for sname, fn in stats.items():
        row = {"stat": sname}
        for k, vals in nums.items():
            row[k] = fn(vals)
        out.append(row)
    return out

def _write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    rows = _summarize_rows_numeric(rows)
    if not rows: 
        return
    # stable union of keys
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def _write_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write arbitrary dict rows to CSV (used for merged per-prompt detail)."""
    if not rows:
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def _lighten(hex_color: str, factor: float):
    r,g,b = _hex_to_rgb(hex_color)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return _rgb_to_hex((r,g,b))

def plot_mlp_steps_ddiff(means, cis, out_path):
    # means/cis ordered as: ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]
    ft_green = "#0b5d1e"
    labels = ["RES","UP\n(after pre)","GATE_PRE\n(before nonlin)","GATE_ACT\n(after nonlin)","POST\n(before down)","DOWN\n(after down)"]
    plt.figure(figsize=(11.5, 4.8))
    plt.bar(labels, means, yerr=cis, color=_lighten(ft_green, 0.20), alpha=0.95, edgecolor="none")
    plt.axhline(0.0, color="#6e6e6e", linewidth=0.9)
    plt.ylabel("ΔΔ L2 (FT−BASE)")
    plt.title("Diff-of-diffs across MLP micro-steps (L2 distance; negative = FT closer)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()

def plot_mlp_paths_individual(vec_fn, outdir, layer_idx):
    ft_green = "#0b5d1e"
    # UP path view
    steps_up   = ["RES","UP","POST","DOWN"]
    ddiff_up   = [vec_fn(f"ddiff_l2_{s}") for s in steps_up]
    means_up   = [float(np.mean(v)) for v in ddiff_up]
    cis_up     = [mean_ci(v)[1] for v in ddiff_up]
    plt.figure(figsize=(7.6,4.0))
    plt.bar(steps_up, means_up, yerr=cis_up, color=_lighten(ft_green, 0.20), alpha=0.95, edgecolor="none")
    plt.axhline(0.0, color="#6e6e6e", linewidth=0.9)
    plt.ylabel("ΔΔ L2 (FT−BASE)")
    plt.title(f"UP path — layer {layer_idx} (negative = FT closer)")
    plt.tight_layout(); plt.savefig(outdir / "mlp_up_path_ddiff_bars.png", dpi=170); plt.close()

    # GATE branch view
    steps_gate = ["GATE_PRE","GATE_ACT"]
    ddiff_gate = [vec_fn(f"ddiff_l2_{s}") for s in steps_gate]
    means_gate = [float(np.mean(v)) for v in ddiff_gate]
    cis_gate   = [mean_ci(v)[1] for v in ddiff_gate]
    plt.figure(figsize=(6.5,4.0))
    plt.bar(steps_gate, means_gate, yerr=cis_gate, color=_lighten(ft_green, 0.20), alpha=0.95, edgecolor="none")
    plt.axhline(0.0, color="#6e6e6e", linewidth=0.9)
    plt.ylabel("ΔΔ L2 (FT−BASE)")
    plt.title(f"GATE branch — layer {layer_idx} (negative = FT closer)")
    plt.tight_layout(); plt.savefig(outdir / "mlp_gate_branch_ddiff_bars.png", dpi=170); plt.close()

def plot_mlp_steps_ddiff_bars_single(means, cis, title, out_path):
    ft_green = "#0b5d1e"
    labels = ["RES","UP\n(after pre)","GATE_PRE\n(before nonlin)","GATE_ACT\n(after nonlin)","POST\n(before down)","DOWN\n(after down)"]
    plt.figure(figsize=(10.5, 4.6))
    plt.bar(labels, means, yerr=cis, color=_lighten(ft_green, 0.20), alpha=0.95, edgecolor="none")
    plt.axhline(0.0, color="#6e6e6e", linewidth=0.9)
    plt.ylabel("ΔΔ L2 (FT−BASE)")
    plt.title(title + " (negative = FT closer)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()

def plot_mlp_steps_ddiff_slices_grid(vec_fn, num_slices: int, outdir: Path):
    """
    Builds one panel per slice + saves per-slice figures.
    Reads ddiff arrays with keys ddiff_l2_{STEP}_S{slice}.
    """
    steps6 = ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]
    # Save per-slice files and aggregate for grid
    per_means = []
    per_cis   = []
    for si in range(1, num_slices+1):
        vals6 = [vec_fn(f"ddiff_l2_{st}_S{si}") for st in steps6]
        means = [float(np.mean(v)) if v.size > 0 else float("nan") for v in vals6]
        cis   = [mean_ci(v)[1] for v in vals6]
        per_means.append(means); per_cis.append(cis)
        plot_mlp_steps_ddiff_bars_single(
            means, cis,
            title=f"Six-step ΔΔ L2 by token slice {si}/{num_slices}",
            out_path=outdir / f"mlp_steps_ddiff_bars_slice{si:02d}.png",
        )

    # Grid figure
    cols = min(3, num_slices)
    rows = int(math.ceil(num_slices / cols))
    ft_green = "#0b5d1e"
    labels = ["RES","UP","G_PRE","G_ACT","POST","DOWN"]  # compact for grid
    plt.figure(figsize=(min(16, 5.0*cols), 3.6*rows))
    for i in range(num_slices):
        ax = plt.subplot(rows, cols, i+1)
        ax.bar(labels, per_means[i], yerr=per_cis[i], color=_lighten(ft_green, 0.20), alpha=0.95, edgecolor="none")
        ax.axhline(0.0, color="#6e6e6e", linewidth=0.9)
        ax.set_title(f"Slice {i+1}/{num_slices}", fontsize=10)
        if i % cols == 0:
            ax.set_ylabel("ΔΔ L2 (FT−BASE)")
        ax.tick_params(axis="x", labelrotation=20)
    plt.suptitle("Six-step ΔΔ L2 by token slices", y=0.995, fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(outdir / "mlp_steps_ddiff_bars_slices_grid.png", dpi=170)
    plt.close()

def plot_jacobian_up_bars(df_base, df_ft, out_path):
    import numpy as np
    import matplotlib.pyplot as plt

    base_color = "#6e6e6e"       # gray
    ft_color   = "#0b5d1e"       # forest green

    val_var_base = float(np.nanmean(df_base["jac_VAR_up_mean"].values))
    val_var_ft   = float(np.nanmean(df_ft["jac_VAR_up_mean"].values))
    val_mean_base= float(np.nanmean(df_base["jac_MEAN_up"].values))
    val_mean_ft  = float(np.nanmean(df_ft["jac_MEAN_up"].values))
    val_rnd_base = float(np.nanmean(df_base["jac_RND_up_mean"].values))
    val_rnd_ft   = float(np.nanmean(df_ft["jac_RND_up_mean"].values))

    groups = ["VAR", "MEAN", "RANDOM"]
    base_vals = [val_var_base, val_mean_base, val_rnd_base]
    ft_vals   = [val_var_ft,   val_mean_ft,   val_rnd_ft]

    x = np.arange(len(groups))
    w = 0.35
    shades = {"VAR":0.25, "MEAN":0.05, "RANDOM":0.45}

    plt.figure(figsize=(7.6, 4.6))
    for i, g in enumerate(groups):
        shade = shades[g]
        bc = _lighten(base_color, shade)
        fc = _lighten(ft_color,   shade*0.8)
        plt.bar(x[i]-w/2, base_vals[i], width=w, color=bc, edgecolor="black", label="BASE" if i==0 else None)
        plt.bar(x[i]+w/2, ft_vals[i],   width=w, color=fc, edgecolor="none",  label="FT"   if i==0 else None)

    plt.xticks(x, groups)
    plt.ylabel("Avg Jacobian norm")
    plt.title("Jacobian (UP) — mean / variance / random\nBASE (gray, black edge) vs FT (forest green)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()

def plot_jacobian_down_bars(df_base, df_ft, out_path):
    import numpy as np
    import matplotlib.pyplot as plt

    base_color = "#6e6e6e"
    ft_color   = "#0b5d1e"

    val_var_base = float(np.nanmean(df_base["jac_VAR_down_mean"].values))
    val_var_ft   = float(np.nanmean(df_ft["jac_VAR_down_mean"].values))
    val_mean_base= float(np.nanmean(df_base["jac_MEAN_down"].values))
    val_mean_ft  = float(np.nanmean(df_ft["jac_MEAN_down"].values))
    val_rnd_base = float(np.nanmean(df_base["jac_RND_down_mean"].values))
    val_rnd_ft   = float(np.nanmean(df_ft["jac_RND_down_mean"].values))

    groups = ["VAR", "MEAN", "RANDOM"]
    base_vals = [val_var_base, val_mean_base, val_rnd_base]
    ft_vals   = [val_var_ft,   val_mean_ft,   val_rnd_ft]

    x = np.arange(len(groups))
    w = 0.35
    shades = {"VAR":0.25, "MEAN":0.05, "RANDOM":0.45}

    plt.figure(figsize=(7.6, 4.6))
    for i, g in enumerate(groups):
        shade = shades[g]
        bc = _lighten(base_color, shade)
        fc = _lighten(ft_color,   shade*0.8)
        plt.bar(x[i]-w/2, base_vals[i], width=w, color=bc, edgecolor="black", label="BASE" if i==0 else None)
        plt.bar(x[i]+w/2, ft_vals[i],   width=w, color=fc, edgecolor="none",  label="FT"   if i==0 else None)

    plt.xticks(x, groups)
    plt.ylabel("Avg Jacobian norm")
    plt.title("Jacobian (DOWN) — mean / variance / random\nBASE (gray, black edge) vs FT (forest green)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()

def plot_jacobian_updown_combo(df_base, df_ft, out_path):
    import numpy as np
    import matplotlib.pyplot as plt

    base_color = "#6e6e6e"
    ft_color   = "#0b5d1e"

    vals = dict(
        up_var_base   = float(np.nanmean(df_base["jac_VAR_up_mean"].values)),
        up_var_ft     = float(np.nanmean(df_ft["jac_VAR_up_mean"].values)),
        up_mean_base  = float(np.nanmean(df_base["jac_MEAN_up"].values)),
        up_mean_ft    = float(np.nanmean(df_ft["jac_MEAN_up"].values)),
        up_rnd_base   = float(np.nanmean(df_base["jac_RND_up_mean"].values)),
        up_rnd_ft     = float(np.nanmean(df_ft["jac_RND_up_mean"].values)),

        down_var_base = float(np.nanmean(df_base["jac_VAR_down_mean"].values)),
        down_var_ft   = float(np.nanmean(df_ft["jac_VAR_down_mean"].values)),
        down_mean_base= float(np.nanmean(df_base["jac_MEAN_down"].values)),
        down_mean_ft  = float(np.nanmean(df_ft["jac_MEAN_down"].values)),
        down_rnd_base = float(np.nanmean(df_base["jac_RND_down_mean"].values)),
        down_rnd_ft   = float(np.nanmean(df_ft["jac_RND_down_mean"].values)),
    )

    left = ["up_var_base","up_var_ft","up_mean_base","up_mean_ft","up_rnd_base","up_rnd_ft"]
    right= ["down_var_base","down_var_ft","down_mean_base","down_mean_ft","down_rnd_base","down_rnd_ft"]

    labels_left  = ["VAR BASE","VAR FT","MEAN BASE","MEAN FT","RANDOM BASE","RANDOM FT"]
    labels_right = ["VAR BASE","VAR FT","MEAN BASE","MEAN FT","RANDOM BASE","RANDOM FT"]

    def bar_style(key):
        is_ft = key.endswith("_ft")
        is_base = not is_ft
        color = ft_color if is_ft else base_color
        edgecolor = "black" if is_base else "none"
        lw = 1.0 if is_base else 0.0
        if "mean" in key:
            hatch = "///"
            shade = 0.05
        elif "var" in key:
            hatch = None
            shade = 0.20
        else:
            hatch = None
            shade = 0.45
        return _lighten(color, shade), edgecolor, lw, hatch

    plt.figure(figsize=(12.0, 4.8))
    n = 6
    x_left = np.arange(n)
    gap = 1.5
    x_right = x_left + n + gap

    for i, key in enumerate(left):
        c, ec, lw, h = bar_style(key)
        plt.bar(x_left[i], vals[key], color=c, edgecolor=ec, linewidth=lw, hatch=h)

    for i, key in enumerate(right):
        c, ec, lw, h = bar_style(key)
        plt.bar(x_right[i], vals[key], color=c, edgecolor=ec, linewidth=lw, hatch=h)

    ticks = list(x_left) + list(x_right)
    tick_labels = labels_left + labels_right
    plt.xticks(ticks, tick_labels, rotation=25)

    mid_left  = (x_left[0] + x_left[-1]) / 2
    mid_right = (x_right[0] + x_right[-1]) / 2
    ymax = plt.gca().get_ylim()[1]
    plt.text(mid_left,  ymax*1.02, "UP",   ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.text(mid_right, ymax*1.02, "DOWN", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.axvline(x_left[-1]+0.5, color="k", linestyle=":", linewidth=0.8, alpha=0.6)

    from matplotlib.patches import Patch
    legend_labels = [
        f"UP VAR BASE = {vals['up_var_base']:.3f}",
        f"UP VAR FT   = {vals['up_var_ft']:.3f}",
        f"UP MEAN BASE= {vals['up_mean_base']:.3f}",
        f"UP MEAN FT  = {vals['up_mean_ft']:.3f}",
        f"UP RANDOM BASE = {vals['up_rnd_base']:.3f}",
        f"UP RANDOM FT   = {vals['up_rnd_ft']:.3f}",
        f"DOWN VAR BASE = {vals['down_var_base']:.3f}",
        f"DOWN VAR FT   = {vals['down_var_ft']:.3f}",
        f"DOWN MEAN BASE= {vals['down_mean_base']:.3f}",
        f"DOWN MEAN FT  = {vals['down_mean_ft']:.3f}",
        f"DOWN RANDOM BASE = {vals['down_rnd_base']:.3f}",
        f"DOWN RANDOM FT   = {vals['down_rnd_ft']:.3f}",
    ]
    leg_elems = []
    all_keys = left + right
    for i, key in enumerate(all_keys):
        color, edgecolor, _, hatch = bar_style(key)
        leg_elems.append(Patch(facecolor=color, edgecolor=edgecolor, hatch=hatch, label=legend_labels[i]))

    plt.legend(handles=leg_elems, loc="upper right", ncol=1, frameon=False)
    plt.ylabel("Avg Jacobian norm")
    plt.title("Jacobian — UP (left) vs DOWN (right)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close()

def plot_jacobian_updown_pair_diffs(df_base, df_ft, out_path):
    """
    Plot the difference (FT - BASE) for Jacobian norms, for UP and DOWN,
    separately for VAR / MEAN / RANDOM directions.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    ft_green = "#0b5d1e"

    diffs_up = [
        float(np.nanmean(df_ft["jac_VAR_up_mean"].values)  - np.nanmean(df_base["jac_VAR_up_mean"].values)),
        float(np.nanmean(df_ft["jac_MEAN_up"].values)      - np.nanmean(df_base["jac_MEAN_up"].values)),
        float(np.nanmean(df_ft["jac_RND_up_mean"].values)  - np.nanmean(df_base["jac_RND_up_mean"].values)),
    ]
    diffs_down = [
        float(np.nanmean(df_ft["jac_VAR_down_mean"].values)  - np.nanmean(df_base["jac_VAR_down_mean"].values)),
        float(np.nanmean(df_ft["jac_MEAN_down"].values)      - np.nanmean(df_base["jac_MEAN_down"].values)),
        float(np.nanmean(df_ft["jac_RND_down_mean"].values)  - np.nanmean(df_base["jac_RND_down_mean"].values)),
    ]

    labels = ["VAR","MEAN","RANDOM"]
    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(8.8, 4.6))
    # UP on left positions, DOWN on right positions
    plt.bar(x - w/2, diffs_up,   width=w, label="UP (FT−BASE)",   color=_lighten(ft_green, 0.20), edgecolor="none")
    plt.bar(x + w/2, diffs_down, width=w, label="DOWN (FT−BASE)", color=_lighten(ft_green, 0.45), edgecolor="none")
    plt.axhline(0.0, color="#6e6e6e", linewidth=0.9)
    plt.xticks(x, labels)
    plt.ylabel("Δ Jacobian norm (FT−BASE)")
    plt.title("Jacobian differences by direction type (UP vs DOWN)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()

def paired_scatter(x, y, out_path, xlabel, ylabel, title):
    import numpy as np
    import matplotlib.pyplot as plt
    base_gray = "#6e6e6e"
    ft_green  = "#0b5d1e"
    x = np.asarray(x); y = np.asarray(y)
    plt.figure(figsize=(6.0,6.0))
    plt.scatter(x, y, s=14, alpha=0.6, color=ft_green)
    mn = min(np.min(x), np.min(y)); mx = max(np.max(x), np.max(y))
    pad = 0.05*(mx-mn if mx>mn else 1.0)
    a = mn - pad; b = mx + pad
    plt.plot([a,b],[a,b], linestyle="--", linewidth=1.0, color=base_gray, alpha=0.9, label="y = x")
    if np.isfinite(x).all() and np.isfinite(y).all() and len(x) > 1:
        xm = x - x.mean(); ym = y - y.mean()
        r = float((xm*ym).sum() / (np.sqrt((xm*xm).sum()) * np.sqrt((ym*ym).sum()) + 1e-12))
        plt.text(0.02, 0.98, f"r = {r:.3f}\nn = {len(x)}", transform=plt.gca().transAxes,
                 ha="left", va="top", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=base_gray, alpha=0.8))
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=170); plt.close()

def build_models_and_tokenizers(
    base_model_name_or_path: str,
    ft_model_name_or_path: Optional[str],
    ft_lora_adapter: Optional[str],
    device: torch.device,
):
    tokenizer_base = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tokenizer_base.pad_token is None:
        tokenizer_base.pad_token = tokenizer_base.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    base = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device).eval()

    if ft_lora_adapter is not None:
        host_name = ft_model_name_or_path or base_model_name_or_path
        ft_host = AutoModelForCausalLM.from_pretrained(host_name, torch_dtype=dtype).to(device).eval()
        if _HAS_PEFT:
            ft = PeftModel.from_pretrained(ft_host, ft_lora_adapter).to(device).eval()
        else:
            raise RuntimeError("peft not available but --ft_lora_adapter provided.")
    else:
        if ft_model_name_or_path is None:
            raise ValueError("Provide --ft_model_name_or_path (merged FT) OR --ft_lora_adapter (LoRA adapter dir).")
        ft = AutoModelForCausalLM.from_pretrained(ft_model_name_or_path, torch_dtype=dtype).to(device).eval()

    tokenizer_ft = AutoTokenizer.from_pretrained(ft_model_name_or_path or base_model_name_or_path, use_fast=True)
    if tokenizer_ft.pad_token is None:
        tokenizer_ft.pad_token = tokenizer_ft.eos_token

    return (base, tokenizer_base), (ft, tokenizer_ft)

def main():
    ap = argparse.ArgumentParser(description="Diff-of-diffs across MLP stations + Jacobian overlays (L2 micro-steps)")
    ap.add_argument("--selection_jsonl", type=str, required=True, help="From SCRIPT A sampler")
    ap.add_argument("--base_model_name_or_path", type=str, required=True)
    ap.add_argument("--ft_model_name_or_path", type=str, default=None)
    ap.add_argument("--ft_lora_adapter", type=str, default=None)
    ap.add_argument("--layer_index", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--max_paraphrases", type=int, default=None)

    ap.add_argument("--compute_jacobians", type=int, default=1, help="0/1")
    ap.add_argument("--jacobian_mode", type=str, default="pca", choices=["pca","raw"])
    ap.add_argument("--topk_pca", type=int, default=8)
    ap.add_argument("--eps_jacobian", type=float, default=1e-3)
    ap.add_argument(
        "--token_slices",
        type=int,
        default=0,
        help="If >0, split each sequence into this many equal token-position slices and compute per-slice six-step ΔΔ figures."
    )

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
        datefmt=":%H:%M:%S",
    )
    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    logging.info("Reading selection: %s", args.selection_jsonl)
    items = load_selection_jsonl(args.selection_jsonl, max_prompts=args.max_prompts)
    logging.info("Loaded %d prompts from selection.", len(items))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (base, tok_base), (ft, tok_ft) = build_models_and_tokenizers(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, device
    )

    logging.info("Models ready. Layer index = %d", args.layer_index)

    logging.info("Capturing MLP stations: BASE")
    base_res = capture_stations_for_model(base, tok_base, items, args.layer_index, device,
                                          args.max_prompts, args.max_paraphrases, token_slices=args.token_slices)

    logging.info("Capturing MLP stations: FT")
    ft_res = capture_stations_for_model(ft, tok_ft, items, args.layer_index, device,
                                        args.max_prompts, args.max_paraphrases, token_slices=args.token_slices)

    _write_summary_csv(base_res["per_prompt"], outdir / f"BASE_mlp_station_metrics_layer{args.layer_index}.csv")
    _write_summary_csv(ft_res["per_prompt"],   outdir / f"FT_mlp_station_metrics_layer{args.layer_index}.csv")

    bmap = {r["prompt_count"]: r for r in base_res["per_prompt"]}
    fmap = {r["prompt_count"]: r for r in ft_res["per_prompt"]}
    merged_rows = []
    for pc in sorted(set(bmap.keys()) & set(fmap.keys())):
        rb, rf = bmap[pc], fmap[pc]
        row = {"prompt_count": pc, "N": rb["N"]}
        for k in [
            "cos_RES","cos_UP","cos_GATE_PRE","cos_GATE_ACT","cos_POST","cos_DOWN",
            "dist_RES","dist_UP","dist_GATE_PRE","dist_GATE_ACT","dist_POST","dist_DOWN",
            "l2_RES","l2_UP","l2_GATE_PRE","l2_GATE_ACT","l2_POST","l2_DOWN",
            "delta_dist_POST_minus_UP","delta_dist_DOWN_minus_UP"
        ]:
            row[k + "_BASE"] = rb.get(k, float("nan")); row[k + "_FT"] = rf.get(k, float("nan"))

        for s in ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]:
            row[f"ddiff_dist_{s}"] = row[f"dist_{s}_FT"] - row[f"dist_{s}_BASE"]
            row[f"ddiff_cos_{s}"]  = row[f"cos_{s}_FT"]  - row[f"cos_{s}_BASE"]
            row[f"ddiff_l2_{s}"]   = row[f"l2_{s}_FT"]   - row[f"l2_{s}_BASE"]

        row["ddiff_delta_POST_minus_UP"] = row["delta_dist_POST_minus_UP_FT"] - row["delta_dist_POST_minus_UP_BASE"]
        row["ddiff_delta_DOWN_minus_UP"] = row["delta_dist_DOWN_minus_UP_FT"] - row["delta_dist_DOWN_minus_UP_BASE"]

        # Per-slice ΔΔ
        S = int(args.token_slices) if args.token_slices and args.token_slices > 0 else 0
        if S > 0:
            stages6 = ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]
            for si in range(1, S+1):
                for st in stages6:
                    # copy BASE/FT (may be NaN if missing)
                    base_key = f"dist_{st}_S{si}"
                    row[base_key + "_BASE"] = rb.get(base_key, float("nan"))
                    row[base_key + "_FT"]   = rf.get(base_key, float("nan"))
                    # ddiff for this slice (dist)
                    row[f"ddiff_dist_{st}_S{si}"] = row[base_key + "_FT"] - row[base_key + "_BASE"]
                    # L2 per-slice
                    l2_key = f"l2_{st}_S{si}"
                    row[l2_key + "_BASE"] = rb.get(l2_key, float("nan"))
                    row[l2_key + "_FT"]   = rf.get(l2_key, float("nan"))
                    row[f"ddiff_l2_{st}_S{si}"] = row[l2_key + "_FT"] - row[l2_key + "_BASE"]

        merged_rows.append(row)

    _write_rows_csv(merged_rows, outdir / f"merged_mlp_station_metrics_layer{args.layer_index}.csv")

    merged = merged_rows
    summaries = {}
    def vec(key): return np.array([r[key] for r in merged if np.isfinite(r[key])], dtype=float)

    for s in ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]:
        dd = vec(f"ddiff_dist_{s}"); cc = vec(f"ddiff_cos_{s}")
        summaries[f"ddiff_dist_{s}"] = summarize_vec(dd)
        summaries[f"ddiff_cos_{s}"]  = summarize_vec(cc)

    for k in ["ddiff_delta_POST_minus_UP","ddiff_delta_DOWN_minus_UP"]:
        summaries[k] = summarize_vec(vec(k))

    for s in ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]:
        b = vec(f"dist_{s}_BASE"); f = vec(f"dist_{s}_FT")
        summaries[f"paired_p_base_vs_ft_dist_{s}"] = float(ttest_paired(f, b))

    with open(outdir / "summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)

    # Figure
    logging.info("Plotting station ΔΔ bars")
    stages = ["RES","UP","POST","DOWN"]
    ddiff_dict = {s: vec(f"ddiff_dist_{s}") for s in stages}
    means = [float(np.mean(ddiff_dict[s])) for s in stages]
    cis   = [mean_ci(ddiff_dict[s])[1] for s in stages]
    ft_green = "#0b5d1e"
    plt.figure(figsize=(8.4,4.2))
    plt.bar(stages, means, yerr=cis, color=_lighten(ft_green, 0.20), alpha=0.95, edgecolor="none")
    plt.axhline(0.0, color="#6e6e6e", linewidth=0.9)
    plt.ylabel("ΔΔ (FT−BASE) of (1−cos)")
    plt.title(f"Diff-of-diffs across MLP stations — layer {args.layer_index}\n(negative = FT closer / denoising)")
    plt.tight_layout(); plt.savefig(outdir / "ddiff_dist_by_station_bars.png", dpi=170); plt.close()

    logging.info("Plotting stage delta ΔΔ bars")
    deltas = ["ddiff_delta_POST_minus_UP","ddiff_delta_DOWN_minus_UP"]
    vals = [vec(k) for k in deltas]
    means = [float(np.mean(v)) for v in vals]
    cis   = [mean_ci(v)[1] for v in vals]
    ft_green = "#0b5d1e"
    plt.figure(figsize=(7.6,4.0))
    plt.bar(["POST−UP", "DOWN−UP"], means, yerr=cis, color=_lighten(ft_green, 0.20), alpha=0.95, edgecolor="none")
    plt.axhline(0.0, color="#6e6e6e", linewidth=0.9)
    plt.ylabel("ΔΔ of stage delta in (1−cos)")
    plt.title(f"Stage deltas (POST−UP, DOWN−UP) — ΔΔ (FT−BASE), layer {args.layer_index}")
    plt.tight_layout(); plt.savefig(outdir / "stage_delta_ddiff_bars.png", dpi=170); plt.close()

    # Six-step micro-view combined + individual path views (NOW L2)
    logging.info("Plotting MLP micro-step ΔΔ bars (six steps, L2)")
    steps6 = ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]
    vals6  = [vec(f"ddiff_l2_{s}") for s in steps6]
    means6 = [float(np.mean(v)) for v in vals6]
    cis6   = [mean_ci(v)[1] for v in vals6]
    plot_mlp_steps_ddiff(means6, cis6, outdir / "mlp_steps_ddiff_bars.png")

    # Helpful breakdowns (L2-based)
    plot_mlp_paths_individual(vec, outdir, args.layer_index)

    # Token-slice six-step ΔΔ figures (L2)
    if args.token_slices and args.token_slices > 0:
        logging.info("Plotting token-slice six-step ΔΔ bars (%d slices)", args.token_slices)
        plot_mlp_steps_ddiff_slices_grid(vec, args.token_slices, outdir)

    logging.info("Plotting BASE vs FT stage-delta scatter")
    x = vec("delta_dist_POST_minus_UP_BASE"); y = vec("delta_dist_POST_minus_UP_FT")
    paired_scatter(x, y, outdir / "base_vs_ft_scatter_deltas.png",
                   xlabel="BASE Δ(POST−UP) in (1−cos)",
                   ylabel="FT Δ(POST−UP) in (1−cos)",
                   title=f"BASE vs FT: Stage delta Δ(POST−UP), layer {args.layer_index}")

    if args.compute_jacobians:
        logging.info("Computing Jacobians — BASE")
        jac_base = jacobian_norms_for_model(
            base, tok_base, items, args.layer_index, device,
            args.topk_pca, args.jacobian_mode,
            args.max_prompts, args.max_paraphrases, eps=args.eps_jacobian
        )
        logging.info("Computing Jacobians — FT")
        jac_ft = jacobian_norms_for_model(
            ft, tok_ft, items, args.layer_index, device,
            args.topk_pca, args.jacobian_mode,
            args.max_prompts, args.max_paraphrases, eps=args.eps_jacobian
        )

        _write_summary_csv(jac_base["per_prompt"], outdir / f"jacobian_BASE_layer{args.layer_index}.csv")
        _write_summary_csv(jac_ft["per_prompt"],   outdir / f"jacobian_FT_layer{args.layer_index}.csv")

        import pandas as pd
        dfB = pd.DataFrame(jac_base["per_prompt"])
        dfF = pd.DataFrame(jac_ft["per_prompt"])

        logging.info("Plotting Jacobian UP / DOWN / COMBO bars")
        plot_jacobian_up_bars(dfB, dfF, outdir / "jacobian_up_bars.png")
        plot_jacobian_down_bars(dfB, dfF, outdir / "jacobian_down_bars.png")
        plot_jacobian_updown_combo(dfB, dfF, outdir / "jacobian_updown_combo_bars.png")
        plot_jacobian_updown_pair_diffs(dfB, dfF, outdir / "jacobian_updown_pair_diffs.png")

    logging.info("All done. Outputs in %s", outdir)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
