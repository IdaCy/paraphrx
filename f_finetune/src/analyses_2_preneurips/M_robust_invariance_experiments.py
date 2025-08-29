#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import math
import random
import logging
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # for log_softmax
import pandas as pd
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Utils & Logging

def setup_logging(outdir: str, level=logging.INFO) -> None:
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "run.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )
    logging.info("Logging to %s", log_path)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Data

@dataclass
class Paraphrase:
    key: str
    text: str

@dataclass
class PromptGroup:
    prompt_count: int
    original: str
    inp: str
    paraphrases: List[Paraphrase]

def read_jsonl(path: str,
               max_prompts: Optional[int] = None,
               max_paraphrases: Optional[int] = None,
               require_input: Optional[bool] = None) -> List[PromptGroup]:
    out: List[PromptGroup] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pc = int(obj["prompt_count"])
            orig = str(obj["instruction_original"])
            inp  = obj.get("input", "")
            if require_input is not None:
                if require_input and not inp:
                    continue
                if not require_input and inp:
                    continue
            paras = obj.get("paraphrases", [])
            ps = [Paraphrase(key=p["key"], text=p["text"]) for p in paras]
            if max_paraphrases is not None and len(ps) > max_paraphrases:
                ps = ps[:max_paraphrases]
            out.append(PromptGroup(prompt_count=pc, original=orig, inp=inp, paraphrases=ps))
            if max_prompts is not None and len(out) >= max_prompts:
                break
    return out

def build_text(instruction: str, inp: str, include_input_tag: bool = True) -> str:
    if inp and include_input_tag:
        return f"{instruction}\n\nInput: {inp}"
    return instruction

# Model Load

@dataclass
class ModelBundle:
    model: AutoModelForCausalLM
    tok: AutoTokenizer
    device: torch.device
    dtype: torch.dtype
    name: str

def load_model(model_path: str,
               lora_path: Optional[str] = None,
               merge_lora: bool = True,
               device: Optional[str] = None,
               dtype: str = "auto",
               attn_implementation: Optional[str] = None,
               name: str = "MODEL") -> ModelBundle:
    logging.info("Loading tokenizer from %s", model_path)
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if dtype == "auto":
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    logging.info("Loading model from %s (dtype=%s)", model_path, torch_dtype)
    kwargs = {"torch_dtype": torch_dtype}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.config.use_cache = False
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()

    if lora_path:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft not installed but lora_path was specified")
        logging.info("Loading LoRA adapter from %s (merge=%s)", lora_path, merge_lora)
        model = PeftModel.from_pretrained(model, lora_path)
        if merge_lora:
            model = model.merge_and_unload()

    return ModelBundle(model=model, tok=tok, device=device_obj, dtype=torch_dtype, name=name)

# Layer-6 Hooks

@dataclass
class StationCaches:
    RES: Optional[torch.Tensor] = None
    UP: Optional[torch.Tensor] = None
    GATE_PRE: Optional[torch.Tensor] = None
    POST: Optional[torch.Tensor] = None
    DOWN: Optional[torch.Tensor] = None

class Layer6Hook:
    """
    Captures & patches at the chosen layer's MLP.
    We do NOT implement the gate; we capture POST from down_proj's *input*.
    """
    def __init__(self, model, layer_index: int):
        self.model = model
        self.layer_index = layer_index
        self.block, self.mlp = self._locate_block_and_mlp(model, layer_index)
        self.up_proj = getattr(self.mlp, "up_proj", None)
        self.gate_proj = getattr(self.mlp, "gate_proj", None)
        self.down_proj = getattr(self.mlp, "down_proj", None)
        if any(x is None for x in [self.up_proj, self.gate_proj, self.down_proj]):
            raise RuntimeError(f"Could not find up_proj/gate_proj/down_proj in mlp at layer {layer_index}")
        self.handles = []
        self.reset()
        # patching
        self.patch_enabled = False
        self.patch_last_token_only = True
        self.patch_batch_index = None
        self.patch_vector = None
        self.patch_mode = "replace"

    def reset(self):
        self.caches = StationCaches()

    def _locate_block_and_mlp(self, model, idx: int):
        for cand in ["model.layers", "gpt_neox.layers", "transformer.h", "layers"]:
            try:
                layers = eval(f"model.{cand}")
                if isinstance(layers, (list, torch.nn.ModuleList)):
                    block = layers[idx]
                    mlp = getattr(block, "mlp", None)
                    if mlp is None:
                        for name in ["feed_forward", "ffn", "ff"]:
                            mlp2 = getattr(block, name, None)
                            if mlp2 is not None:
                                mlp = mlp2
                                break
                    if mlp is None:
                        continue
                    return block, mlp
            except Exception:
                continue
        raise RuntimeError(f"Could not locate transformer blocks and MLP at index {idx}")

    def _hook_mlp_input(self, module, inputs):
        self.caches.RES = inputs[0].detach()

    def _hook_up(self, module, inputs, output):
        self.caches.UP = output.detach()

    def _hook_gate_pre(self, module, inputs, output):
        self.caches.GATE_PRE = output.detach()

    def _hook_down_pre(self, module, inputs):
        self.caches.POST = inputs[0].detach()

    def _hook_mlp_output(self, module, inputs, output):
        out = output
        if self.patch_enabled and (self.patch_vector is not None) and (self.patch_batch_index is not None):
            out = out.clone()
            if self.patch_last_token_only:
                out[self.patch_batch_index, -1, :] = (
                    self.patch_vector if self.patch_mode == "replace"
                    else out[self.patch_batch_index, -1, :] + self.patch_vector
                )
            else:
                out[self.patch_batch_index, :, :] = (
                    self.patch_vector if self.patch_mode == "replace"
                    else out[self.patch_batch_index, :, :] + self.patch_vector
                )
        self.caches.DOWN = out.detach()
        return out

    def install(self):
        self.handles.append(self.mlp.register_forward_pre_hook(self._hook_mlp_input, with_kwargs=False))
        self.handles.append(self.up_proj.register_forward_hook(self._hook_up))
        self.handles.append(self.gate_proj.register_forward_hook(self._hook_gate_pre))
        self.handles.append(self.down_proj.register_forward_pre_hook(self._hook_down_pre, with_kwargs=False))
        self.handles.append(self.mlp.register_forward_hook(self._hook_mlp_output))

    def remove(self):
        for h in self.handles:
            try: h.remove()
            except Exception: pass
        self.handles.clear()

# Tokenization / Batching

def tokenize_batch(tok, texts: List[str], device, max_length: Optional[int] = None):
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    return {k: v.to(device) for k, v in enc.items()}

# Metrics (CPU-safe KL)

def _cpu_f32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(dtype=torch.float32, device="cpu")

def sym_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Symmetric KL on the same device/dtype (caller ensures that)."""
    p = torch.log_softmax(p_logits, dim=-1)
    q = torch.log_softmax(q_logits, dim=-1)
    p_prob = p.exp()
    q_prob = q.exp()
    kl_pq = (p_prob * (p - q)).sum(dim=-1)
    kl_qp = (q_prob * (q - p)).sum(dim=-1)
    return kl_pq + kl_qp

def sym_kl_cpu(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Force both to CPU/fp32 to avoid device/dtype issues."""
    return sym_kl(_cpu_f32(p_logits), _cpu_f32(q_logits))

# Sensitive Subspace

def _center(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    return X - mu, mu

def build_logit_sensitive_basis(
    X: np.ndarray,  # [N, D]
    Y: np.ndarray,  # [N, V']
    k: int = 8,
    lam: float = 1e-3,
) -> np.ndarray:
    """
    SVD ridge (no sklearn/scipy ridge):
      With centered X, Y and λ>0, X = U Σ Vᵀ  ⇒  Wᵀ = V diag(Σ / (Σ² + λ)) Uᵀ Y.
    Returns U_k with shape [D, k] (orthonormal columns).
    """
    lam = float(max(lam, 1e-12))
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    Xc, _ = _center(X); Yc, _ = _center(Y)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)   # U:[N,r], s:[r], Vt:[r,D]
    f = s / (s * s + lam)                               # [r]
    UtY = U.T @ Yc                                      # [r, V']
    UtY *= f[:, None]
    Wt = Vt.T @ UtY                                     # [D, V']
    Uw, _, _ = np.linalg.svd(Wt, full_matrices=False)   # Uw:[D,D']
    k = int(max(1, min(k, Uw.shape[1])))
    U_k, _ = np.linalg.qr(Uw[:, :k])                    # orthonormalize
    return U_k.astype(np.float64)

def project_components(deltaH: np.ndarray, U_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    P = U_k @ U_k.T  # [D, D]
    sens = deltaH @ P.T
    orth = deltaH - sens
    return sens, orth

# Core Runner

@dataclass
class RunConfig:
    jsonl_path: str
    outdir: str
    model_base: str
    model_ft: Optional[str]
    lora_ft: Optional[str]
    merge_lora: bool
    layer_index: int
    max_prompts: Optional[int]
    max_paraphrases: Optional[int]
    max_length: Optional[int]
    batch_size: int
    seed: int
    device: Optional[str]
    dtype: str
    topk_vocab: int
    topk_sv: int
    ridge_lam: float
    include_input_tag: bool
    do_experiment_A: bool
    do_experiment_B: bool
    do_mediation: bool

class RobustInvarianceRunner:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        os.makedirs(cfg.outdir, exist_ok=True)
        setup_logging(cfg.outdir)
        set_seed(cfg.seed)
        # load models
        self.base = load_model(cfg.model_base, None, True, cfg.device, cfg.dtype, name="BASE")
        if cfg.model_ft:
            self.ft = load_model(cfg.model_ft, cfg.lora_ft, cfg.merge_lora, cfg.device, cfg.dtype, name="FT")
        else:
            self.ft = load_model(cfg.model_base, cfg.lora_ft, cfg.merge_lora, cfg.device, cfg.dtype, name="FT")
        # hooks
        self.hook_base = Layer6Hook(self.base.model, cfg.layer_index)
        self.hook_ft   = Layer6Hook(self.ft.model,   cfg.layer_index)
        self.hook_base.install()
        self.hook_ft.install()

    def __del__(self):
        try:
            self.hook_base.remove()
            self.hook_ft.remove()
        except Exception:
            pass

    # Encoding helpers

    def _encode(self, bundle: ModelBundle, texts: List[str]) -> Dict[str, torch.Tensor]:
        return tokenize_batch(bundle.tok, texts, bundle.device, self.cfg.max_length)

    @torch.no_grad()
    def _forward_logits(self, bundle: ModelBundle, batch: Dict[str, torch.Tensor]):
        return bundle.model(**batch, output_attentions=False)

    def _last_logits(self, logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(dim=1) - 1
        out = []
        for i in range(logits.size(0)):
            out.append(logits[i, lengths[i].item(), :].unsqueeze(0))
        return torch.cat(out, dim=0)

    # Capture

    @torch.no_grad()
    def capture_group(self, bundle: ModelBundle, group: PromptGroup) -> Dict[str, Any]:
        texts = [build_text(group.original, group.inp, self.cfg.include_input_tag)] + \
                [build_text(p.text, group.inp, self.cfg.include_input_tag) for p in group.paraphrases]
        keys  = ["instruction_original"] + [p.key for p in group.paraphrases]

        BATCH = self.cfg.batch_size
        logits_last = []
        H_last = {"RES":[], "UP":[], "GATE_PRE":[], "POST":[], "DOWN":[]}

        for start in range(0, len(texts), BATCH):
            chunk = texts[start:start+BATCH]
            enc = self._encode(bundle, chunk)
            out = self._forward_logits(bundle, enc)
            last = self._last_logits(out.logits, enc["attention_mask"])
            logits_last.append(_cpu_f32(last))  # store CPU/fp32

            caches = getattr(self, "hook_"+bundle.name.lower()).caches
            idxs = (enc["attention_mask"].sum(dim=1) - 1).tolist()
            for k in H_last.keys():
                tens = getattr(caches, k, None)
                if tens is None:
                    continue
                tens = tens.detach()  # device of model
                # extract last valid token per item, then move to CPU/fp32
                last_vecs = torch.stack([tens[i, idxs[i], :] for i in range(tens.size(0))], dim=0)
                H_last[k].append(_cpu_f32(last_vecs))

        logits_last = torch.cat(logits_last, dim=0)  # [N,V] on CPU/fp32
        for k in H_last:
            H_last[k] = torch.cat(H_last[k], dim=0) if len(H_last[k]) > 0 else torch.empty((0,0), dtype=torch.float32)

        return {
            "texts": texts,
            "keys": keys,
            "logits_last": logits_last,  # CPU/fp32 [N,V]
            "H_last": H_last,            # dict of CPU/fp32 [N,D]
        }

    # Subspace inputs

    def _prepare_subspace_inputs(self, pack, topk_vocab: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        logits = pack["logits_last"]             # CPU/fp32 [N,V]
        H_last = pack["H_last"]["DOWN"]          # CPU/fp32 [N,D]
        if H_last.ndim != 2 or H_last.shape[0] < 2:
            raise ValueError("Need at least 1 paraphrase (N>=2) to build subspace.")
        N, D = H_last.shape
        H0 = H_last[0:1, :].expand(N-1, -1)      # [N-1,D]
        X = (H_last[1:, :] - H0).numpy()         # [N-1,D]

        topk = int(max(1, min(topk_vocab, logits.shape[-1])))
        top_idx = torch.topk(logits, k=topk, dim=-1).indices  # [N, topk] (still CPU)
        vocab_idx = torch.unique(top_idx.flatten()).cpu().numpy().tolist()
        Y = (logits[1:, vocab_idx] - logits[0:1, vocab_idx]).numpy()  # [N-1, V']

        return X, Y, vocab_idx

    # Experiment A

    @torch.no_grad()
    def experiment_A(self, group: PromptGroup, pack_ft: Dict[str,Any]) -> Dict[str, Any]:
        texts = pack_ft["texts"]
        if len(texts) < 2:
            return {"prompt_count": group.prompt_count, "error": "No paraphrases (N<2)"}

        X, Y, vocab_idx = self._prepare_subspace_inputs(pack_ft, self.cfg.topk_vocab)
        U_k = build_logit_sensitive_basis(X, Y, k=self.cfg.topk_sv, lam=self.cfg.ridge_lam)  # [D,k]

        H_last = pack_ft["H_last"]["DOWN"]       # CPU/fp32 [N,D]
        logits_last = pack_ft["logits_last"]     # CPU/fp32 [N,V]

        base_kl = sym_kl_cpu(logits_last[0:1, :].expand(H_last.size(0)-1, -1),
                             logits_last[1:, :]).cpu().numpy()

        hook = self.hook_ft
        hook.patch_enabled = True
        hook.patch_last_token_only = True
        hook.patch_mode = "replace"

        kl_orth, kl_sens = [], []
        for idx in range(1, len(texts)):
            delta = (H_last[idx] - H_last[0]).numpy()[None, :]   # [1,D]
            sens_np, orth_np = project_components(delta, U_k)    # numpy
            sens = torch.tensor(sens_np[0], device=self.ft.device, dtype=self.ft.dtype)
            orth = torch.tensor(orth_np[0], device=self.ft.device, dtype=self.ft.dtype)
            h0_dev = H_last[0].to(self.ft.device, self.ft.dtype)

            # Orth-only
            hook.patch_batch_index = 0
            hook.patch_vector = (h0_dev + orth)
            enc = self._encode(self.ft, [texts[idx]])
            out = self._forward_logits(self.ft, enc)
            logit_last = self._last_logits(out.logits, enc["attention_mask"])
            kl_orth.append(float(sym_kl_cpu(logits_last[0:1, :], logit_last).item()))

            # Sens-only
            hook.patch_vector = (h0_dev + sens)
            out = self._forward_logits(self.ft, enc)
            logit_last = self._last_logits(out.logits, enc["attention_mask"])
            kl_sens.append(float(sym_kl_cpu(logits_last[0:1, :], logit_last).item()))

        hook.patch_enabled = False
        return {
            "prompt_count": group.prompt_count,
            "base_kl_mean": float(np.mean(base_kl)),
            "orth_kl_mean": float(np.mean(kl_orth)) if kl_orth else float("nan"),
            "sens_kl_mean": float(np.mean(kl_sens)) if kl_sens else float("nan"),
            "base_kl": base_kl.tolist(),
            "orth_kl": kl_orth,
            "sens_kl": kl_sens,
            "U_k": U_k.tolist(),
            "vocab_idx": vocab_idx,
        }

    # Experiment B

    @torch.no_grad()
    def experiment_B(self, group: PromptGroup, pack_base: Dict[str,Any], pack_ft: Dict[str,Any]) -> Dict[str, Any]:
        texts = pack_ft["texts"]
        if len(texts) < 2:
            return {"prompt_count": group.prompt_count, "error": "No paraphrases (N<2)"}

        Xb, Yb, _ = self._prepare_subspace_inputs(pack_base, self.cfg.topk_vocab)
        Ub_np = build_logit_sensitive_basis(Xb, Yb, k=self.cfg.topk_sv, lam=self.cfg.ridge_lam)
        Xf, Yf, _ = self._prepare_subspace_inputs(pack_ft, self.cfg.topk_vocab)
        Uf_np = build_logit_sensitive_basis(Xf, Yf, k=self.cfg.topk_sv, lam=self.cfg.ridge_lam)

        Ub = torch.tensor(Ub_np, device=self.ft.device, dtype=self.ft.dtype)  # [D,k]
        Uf = torch.tensor(Uf_np, device=self.ft.device, dtype=self.ft.dtype)
        Pf = Uf @ Uf.T  # [D,D]

        H_last = pack_ft["H_last"]["DOWN"]       # CPU/fp32 [N,D]
        logits_last = pack_ft["logits_last"]     # CPU/fp32 [N,V]

        base_kl = sym_kl_cpu(logits_last[0:1, :].expand(H_last.size(0)-1, -1),
                             logits_last[1:, :]).cpu().numpy()

        hook = self.hook_ft
        hook.patch_enabled = True
        hook.patch_last_token_only = True
        hook.patch_mode = "replace"

        kl_rot = []
        for idx in range(1, len(texts)):
            h_orig = H_last[0].to(self.ft.device, self.ft.dtype)
            h_para = H_last[idx].to(self.ft.device, self.ft.dtype)
            h_sens = (h_para - h_orig) @ Pf
            h_orth = (h_para - h_orig) - h_sens
            c = Uf.T @ h_sens
            v_rot = Ub @ c
            h_rot = h_orig + h_orth + v_rot
            hook.patch_batch_index = 0
            hook.patch_vector = h_rot
            enc = self._encode(self.ft, [texts[idx]])
            out = self._forward_logits(self.ft, enc)
            logit_last = self._last_logits(out.logits, enc["attention_mask"])
            kl_rot.append(float(sym_kl_cpu(logits_last[0:1, :], logit_last).item()))

        hook.patch_enabled = False
        return {
            "prompt_count": group.prompt_count,
            "ft_base_kl_mean": float(np.mean(base_kl)),
            "swap_rot_kl_mean": float(np.mean(kl_rot)) if kl_rot else float("nan"),
            "ft_base_kl": base_kl.tolist(),
            "swap_rot_kl": kl_rot,
            "Ub": Ub.detach().cpu().numpy().tolist(),
            "Uf": Uf.detach().cpu().numpy().tolist(),
        }

    # Mediation C

    def compute_sensitive_fraction(self, pack, U_k: np.ndarray) -> float:
        H_last = pack["H_last"]["DOWN"].numpy()  # [N,D]
        delta = H_last[1:, :] - H_last[0:1, :]
        sens, _ = project_components(delta, U_k)
        num = float(np.sum(np.sum(sens**2, axis=1)))
        den = float(np.sum(np.sum(delta**2, axis=1)) + 1e-12)
        return num / den

    @torch.no_grad()
    def mediation_metrics_per_prompt(self, pack_base, pack_ft) -> Dict[str, Any]:
        try:
            Xb, Yb, _ = self._prepare_subspace_inputs(pack_base, self.cfg.topk_vocab)
            Ub = build_logit_sensitive_basis(Xb, Yb, k=self.cfg.topk_sv, lam=self.cfg.ridge_lam)
            Xf, Yf, _ = self._prepare_subspace_inputs(pack_ft, self.cfg.topk_vocab)
            Uf = build_logit_sensitive_basis(Xf, Yf, k=self.cfg.topk_sv, lam=self.cfg.ridge_lam)
        except Exception as e:
            return {"error": f"mediation subspace build failed: {e}"}

        try:
            frac_b = self.compute_sensitive_fraction(pack_base, Ub)
            frac_f = self.compute_sensitive_fraction(pack_ft, Uf)

            lb, lf = pack_base["logits_last"], pack_ft["logits_last"]
            kl_b = float(sym_kl_cpu(lb[0:1, :].expand(lb.size(0)-1, -1), lb[1:, :]).mean().item())
            kl_f = float(sym_kl_cpu(lf[0:1, :].expand(lf.size(0)-1, -1), lf[1:, :]).mean().item())

            def _var_and_norm(pack):
                H = pack["H_last"]["RES"].numpy()  # [N,D]
                Z = H - H.mean(axis=0, keepdims=True)
                trace = float(np.sum(np.var(Z, axis=0)))
                mean_norm_sq = float(np.sum((H.mean(axis=0))**2))
                return trace, mean_norm_sq

            tr_b, ms_b = _var_and_norm(pack_base)
            tr_f, ms_f = _var_and_norm(pack_ft)

            return {
                "frac_sens_BASE": frac_b,
                "frac_sens_FT":   frac_f,
                "delta_frac":     (frac_b - frac_f),
                "kl_base":        kl_b,
                "kl_ft":          kl_f,
                "delta_kl":       (kl_b - kl_f),
                "trace_res_BASE": tr_b,
                "trace_res_FT":   tr_f,
                "delta_trace_res": (tr_f - tr_b),
                "mean_normsq_BASE": ms_b,
                "mean_normsq_FT":   ms_f,
                "delta_mean_normsq": (ms_f - ms_b),
            }
        except Exception as e:
            return {"error": f"mediation metrics failed: {e}"}

    # Orchestrate over dataset

    def run(self):
        cfg = self.cfg
        logging.info("Reading data from %s", cfg.jsonl_path)
        groups = read_jsonl(cfg.jsonl_path, cfg.max_prompts, cfg.max_paraphrases, require_input=None)
        logging.info("Loaded %d prompt groups", len(groups))

        rows_A, rows_B, rows_med = [], [], []
        failed_log = []

        for gi, group in enumerate(tqdm(groups, desc="Groups")):
            try:
                self.hook_base.reset(); self.hook_ft.reset()
                pack_base = self.capture_group(self.base, group)
                self.hook_ft.reset()
                pack_ft   = self.capture_group(self.ft, group)

                if cfg.do_experiment_A:
                    try:
                        rows_A.append(self.experiment_A(group, pack_ft))
                    except Exception as e:
                        msg = f"Experiment A failed for prompt_count={group.prompt_count}: {e}"
                        logging.exception(msg)
                        rows_A.append({"prompt_count": group.prompt_count, "error": str(e)})
                        failed_log.append({"prompt_count": group.prompt_count, "stage": "expA", "error": str(e)})

                if cfg.do_experiment_B:
                    try:
                        rows_B.append(self.experiment_B(group, pack_base, pack_ft))
                    except Exception as e:
                        msg = f"Experiment B failed for prompt_count={group.prompt_count}: {e}"
                        logging.exception(msg)
                        rows_B.append({"prompt_count": group.prompt_count, "error": str(e)})
                        failed_log.append({"prompt_count": group.prompt_count, "stage": "expB", "error": str(e)})

                if cfg.do_mediation:
                    med = self.mediation_metrics_per_prompt(pack_base, pack_ft)
                    med["prompt_count"] = group.prompt_count
                    if "error" in med:
                        failed_log.append({"prompt_count": group.prompt_count, "stage": "mediation", "error": med["error"]})
                    rows_med.append(med)

            except Exception as e:
                logging.exception("Capture failed for prompt_count=%s: %s", group.prompt_count, e)
                failed_log.append({"prompt_count": group.prompt_count, "stage": "capture", "error": str(e)})

        # Save tables
        if rows_A:
            dfA = pd.DataFrame(rows_A)
            dfA.to_csv(os.path.join(cfg.outdir, "experiment_A_projection_patch.csv"), index=False)
            logging.info("Saved Experiment A table with %d rows", len(dfA))
            self._plot_experiment_A(dfA, cfg.outdir)
        if rows_B:
            dfB = pd.DataFrame(rows_B)
            dfB.to_csv(os.path.join(cfg.outdir, "experiment_B_subspace_swap.csv"), index=False)
            logging.info("Saved Experiment B table with %d rows", len(dfB))
            self._plot_experiment_B(dfB, cfg.outdir)
        if rows_med:
            dfM = pd.DataFrame(rows_med)
            dfM.to_csv(os.path.join(cfg.outdir, "mediation_metrics.csv"), index=False)
            logging.info("Saved mediation metrics table with %d rows", len(dfM))
            try:
                self._run_mediation_regression(dfM, cfg.outdir)
            except Exception as e:
                logging.exception("Mediation regression failed: %s", e)

        # Save failed rows log
        if failed_log:
            with open(os.path.join(cfg.outdir, "failed_groups.jsonl"), "w", encoding="utf-8") as f:
                for row in failed_log:
                    f.write(json.dumps(row) + "\n")
            logging.info("Wrote %d failure records to failed_groups.jsonl", len(failed_log))

        # Markdown overview (tables of key stats)
        try:
            self._write_markdown_report(
                cfg.outdir,
                cfg,
                n_groups=len(groups),
                dfA=locals().get("dfA", None),
                dfB=locals().get("dfB", None),
                dfM=locals().get("dfM", None),
            )
        except Exception as e:
            logging.exception("Failed to write Markdown overview: %s", e)

        logging.info("Done. Outputs are in %s", cfg.outdir)

    # Plots

    def _ci(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return 0.0
        return 1.96 * (np.nanstd(x, ddof=1) / math.sqrt(np.count_nonzero(~np.isnan(x))))

    def _plot_experiment_A(self, df: pd.DataFrame, outdir: str):
        need = ["base_kl_mean", "orth_kl_mean", "sens_kl_mean"]
        if not all(col in df.columns for col in need):
            logging.warning("Experiment A: not enough valid rows to plot (missing columns).")
            return
        dff = df.dropna(subset=need)
        dff = dff[[c for c in need if c in dff.columns]]
        if dff.empty:
            logging.warning("Experiment A: 0 valid rows after filtering; skipping plot.")
            return
        logging.info("Experiment A: %d rows total; %d valid for plotting; %d failed",
                     len(df), len(dff), len(df) - len(dff))
        means = {
            "FT baseline": dff["base_kl_mean"].astype(float).values,
            "Orth-only (patched)": dff["orth_kl_mean"].astype(float).values,
            "Sens-only (patched)": dff["sens_kl_mean"].astype(float).values,
        }
        labels = list(means.keys())
        vals = [np.nanmean(means[k]) for k in labels]
        cis  = [self._ci(means[k]) for k in labels]
        plt.figure(figsize=(8,5))
        plt.bar(labels, vals, yerr=cis, capsize=4)
        plt.ylabel("Symmetric KL (orig || paraphrase)")
        plt.title("Experiment A: Projection Patching @ DOWN (FT model)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "expA_projection_patch_bars.png"))
        plt.close()

    def _plot_experiment_B(self, df: pd.DataFrame, outdir: str):
        need = ["ft_base_kl_mean", "swap_rot_kl_mean"]
        if not all(col in df.columns for col in need):
            logging.warning("Experiment B: not enough valid rows to plot (missing columns).")
            return
        dff = df.dropna(subset=need)
        dff = dff[[c for c in need if c in dff.columns]]
        if dff.empty:
            logging.warning("Experiment B: 0 valid rows after filtering; skipping plot.")
            return
        logging.info("Experiment B: %d rows total; %d valid for plotting; %d failed",
                     len(df), len(dff), len(df) - len(dff))
        means = {
            "FT baseline": dff["ft_base_kl_mean"].astype(float).values,
            "FT with FT→BASE sensitive-rotation": dff["swap_rot_kl_mean"].astype(float).values,
        }
        labels = list(means.keys())
        vals = [np.nanmean(means[k]) for k in labels]
        cis  = [self._ci(means[k]) for k in labels]
        plt.figure(figsize=(9,5))
        plt.bar(labels, vals, yerr=cis, capsize=4)
        plt.ylabel("Symmetric KL (orig || paraphrase)")
        plt.title("Experiment B: Subspace Swap at DOWN")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "expB_subspace_swap_bars.png"))
        plt.close()

    def _run_mediation_regression(self, dfM: pd.DataFrame, outdir: str):
        import statsmodels.api as sm
        need = ["delta_kl", "delta_frac", "delta_trace_res", "delta_mean_normsq"]
        if not all(c in dfM.columns for c in need):
            logging.warning("Mediation: missing columns; skipping regression.")
            return
        df = dfM.dropna(subset=need)
        if len(df) < 3:
            logging.warning("Mediation: too few rows for regression after filtering; skipping.")
            return
        y = df["delta_kl"].astype(float).values
        X = np.stack([
            df["delta_frac"].astype(float).values,
            df["delta_trace_res"].astype(float).values,
            df["delta_mean_normsq"].astype(float).values,
        ], axis=1)
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        with open(os.path.join(outdir, "mediation_regression.txt"), "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
        # Scatter
        plt.figure(figsize=(6,5))
        xs = df["delta_frac"].astype(float).values
        plt.scatter(xs, y, s=10, alpha=0.7)
        m, b = np.polyfit(xs, y, 1)
        grid = np.linspace(xs.min(), xs.max(), 100)
        plt.plot(grid, m*grid + b, linestyle="--")
        plt.xlabel("Δ sensitive-fraction at DOWN (BASE − FT)")
        plt.ylabel("Δ KL (BASE − FT)")
        plt.title("Mediation: ΔFrac vs ΔKL")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "mediation_scatter_deltafrac_deltakl.png"))
        plt.close()

    def _agg_stats(self, arr: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        n = int(arr.size)
        if n == 0:
            return {"n": 0, "mean": np.nan, "std": np.nan, "ci95": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
        return {
            "n": n,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
            "ci95": float(self._ci(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def _md_row(self, label: str, stats: Dict[str, float]) -> str:
        if stats["n"] == 0:
            return f"| {label} | — | — | — | 0 | — | — |"
        return (f"| {label} | {stats['mean']:.4g} ± {stats['ci95']:.4g} | {stats['median']:.4g} "
                f"| {stats['std']:.4g} | {stats['n']} | {stats['min']:.4g} | {stats['max']:.4g} |")

    def _write_markdown_report(self,
                               outdir: str,
                               cfg: RunConfig,
                               n_groups: int,
                               dfA: Optional[pd.DataFrame],
                               dfB: Optional[pd.DataFrame],
                               dfM: Optional[pd.DataFrame]) -> None:
        md_path = os.path.join(outdir, "results_overview.md")

        # Helper to safely pull columns
        def _vals(df: Optional[pd.DataFrame], col: str) -> np.ndarray:
            if df is None or col not in df.columns:
                return np.array([], dtype=float)
            return pd.to_numeric(df[col], errors="coerce").to_numpy()

        lines = []
        lines.append(f"# Robust Invariance — Statistical Overview\n")
        lines.append(f"- **Data**: `{cfg.jsonl_path}`")
        lines.append(f"- **Outdir**: `{outdir}`")
        lines.append(f"- **Models**: BASE=`{cfg.model_base}`; FT=`{cfg.model_ft or cfg.model_base}`"
                     f"{' + LoRA='+cfg.lora_ft if cfg.lora_ft else ''}")
        lines.append(f"- **Layer index**: {cfg.layer_index}")
        lines.append(f"- **Device/DType**: {cfg.device or 'auto'} / {cfg.dtype}")
        lines.append(f"- **Params**: topk_vocab={cfg.topk_vocab}, topk_sv={cfg.topk_sv}, ridge_lam={cfg.ridge_lam}")
        lines.append(f"- **Groups loaded**: {n_groups}\n")

        # Failures (if any)
        fail_log = os.path.join(outdir, "failed_groups.jsonl")
        if os.path.exists(fail_log):
            by_stage: Dict[str, int] = {}
            with open(fail_log, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        by_stage[obj.get("stage", "unknown")] = by_stage.get(obj.get("stage", "unknown"), 0) + 1
                    except Exception:
                        pass
            if by_stage:
                lines.append("## Failures\n")
                for k, v in sorted(by_stage.items()):
                    lines.append(f"- {k}: {v}")
                lines.append("")

        # Experiment A
        if dfA is not None and any(c in dfA.columns for c in ["base_kl_mean","orth_kl_mean","sens_kl_mean"]):
            lines.append("## Experiment A — Projection Patching @ DOWN (FT)\n")
            dff = dfA.dropna(subset=[c for c in ["base_kl_mean","orth_kl_mean","sens_kl_mean"] if c in dfA.columns])
            lines.append(f"- Rows: total={len(dfA)}, valid={len(dff)}, failed={len(dfA)-len(dff)}\n")

            base = _vals(dff, "base_kl_mean")
            orth = _vals(dff, "orth_kl_mean")
            sens = _vals(dff, "sens_kl_mean")
            d_orth = orth - base if base.size and orth.size else np.array([], dtype=float)
            d_sens = sens - base if base.size and sens.size else np.array([], dtype=float)

            lines.append("| Metric | mean ± 95%CI | median | std | n | min | max |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            lines.append(self._md_row("FT baseline KL", self._agg_stats(base)))
            lines.append(self._md_row("Orth-only KL", self._agg_stats(orth)))
            lines.append(self._md_row("Sens-only KL", self._agg_stats(sens)))
            lines.append(self._md_row("Δ Orth−Base", self._agg_stats(d_orth)))
            lines.append(self._md_row("Δ Sens−Base", self._agg_stats(d_sens)))
            lines.append("")
            # Link plots if present
            expA_img = os.path.join(outdir, "expA_projection_patch_bars.png")
            if os.path.exists(expA_img):
                lines.append(f"![Experiment A](./{os.path.basename(expA_img)})\n")

        # Experiment B
        if dfB is not None and any(c in dfB.columns for c in ["ft_base_kl_mean","swap_rot_kl_mean"]):
            lines.append("## Experiment B — Subspace Swap (FT→BASE) @ DOWN\n")
            dff = dfB.dropna(subset=[c for c in ["ft_base_kl_mean","swap_rot_kl_mean"] if c in dfB.columns])
            lines.append(f"- Rows: total={len(dfB)}, valid={len(dff)}, failed={len(dfB)-len(dff)}\n")

            base = _vals(dff, "ft_base_kl_mean")
            swap = _vals(dff, "swap_rot_kl_mean")
            d_swap = swap - base if base.size and swap.size else np.array([], dtype=float)

            lines.append("| Metric | mean ± 95%CI | median | std | n | min | max |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            lines.append(self._md_row("FT baseline KL", self._agg_stats(base)))
            lines.append(self._md_row("FT with FT→BASE rotation KL", self._agg_stats(swap)))
            lines.append(self._md_row("Δ Swap−Base", self._agg_stats(d_swap)))
            lines.append("")
            expB_img = os.path.join(outdir, "expB_subspace_swap_bars.png")
            if os.path.exists(expB_img):
                lines.append(f"![Experiment B](./{os.path.basename(expB_img)})\n")

        # Mediation
        if dfM is not None and any(c in dfM.columns for c in ["delta_kl","delta_frac","delta_trace_res","delta_mean_normsq"]):
            lines.append("## Mediation — Δ sensitive-fraction vs Δ KL\n")
            dff = dfM.dropna(subset=[c for c in ["delta_kl","delta_frac"] if c in dfM.columns])
            lines.append(f"- Rows: total={len(dfM)}, valid for ΔKL/ΔFrac={len(dff)}\n")

            delta_kl = _vals(dff, "delta_kl")
            delta_frac = _vals(dff, "delta_frac")

            # simple correlation & slope (no extra dependencies)
            if delta_kl.size and delta_frac.size and delta_kl.size == delta_frac.size:
                r = float(np.corrcoef(delta_frac, delta_kl)[0,1])
                m, b = np.polyfit(delta_frac, delta_kl, 1)
                lines.append(f"- Pearson r(ΔFrac, ΔKL) = **{r:.4g}**; linear slope ≈ **{m:.4g}**\n")

            lines.append("| Metric | mean ± 95%CI | median | std | n | min | max |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for col, label in [
                ("frac_sens_BASE","Frac sensitive (BASE)"),
                ("frac_sens_FT","Frac sensitive (FT)"),
                ("delta_frac","ΔFrac (BASE−FT)"),
                ("kl_base","KL BASE"),
                ("kl_ft","KL FT"),
                ("delta_kl","ΔKL (BASE−FT)"),
                ("trace_res_BASE","Trace Var RES (BASE)"),
                ("trace_res_FT","Trace Var RES (FT)"),
                ("delta_trace_res","Δ Trace Var RES"),
                ("mean_normsq_BASE","Mean Norm² RES (BASE)"),
                ("mean_normsq_FT","Mean Norm² RES (FT)"),
                ("delta_mean_normsq","Δ Mean Norm² RES"),
            ]:
                vals = _vals(dfM, col)
                lines.append(self._md_row(label, self._agg_stats(vals)))
            lines.append("")
            med_img = os.path.join(outdir, "mediation_scatter_deltafrac_deltakl.png")
            if os.path.exists(med_img):
                lines.append(f"![Mediation](./{os.path.basename(med_img)})\n")

            # include OLS summary if created
            ols_txt = os.path.join(outdir, "mediation_regression.txt")
            if os.path.exists(ols_txt):
                lines.append("<details><summary>OLS regression summary</summary>\n\n```text")
                try:
                    with open(ols_txt, "r", encoding="utf-8") as f:
                        lines.append(f.read().rstrip())
                except Exception as e:
                    lines.append(f"(could not read OLS summary: {e})")
                lines.append("```\n</details>\n")

        # Write file
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logging.info("Wrote Markdown overview to %s", md_path)

# CLI

def build_argparser():
    p = argparse.ArgumentParser(description="Robust Invariance Experiments — Layer-6 Mechanism Suite (device-safe)")
    p.add_argument("--jsonl", type=str, required=True, help="Path to JSONL data file")
    p.add_argument("--outdir", type=str, required=True, help="Output directory")
    p.add_argument("--model_base", type=str, required=True, help="HF path for BASE model")
    p.add_argument("--model_ft", type=str, default=None, help="HF path for FT model (optional)")
    p.add_argument("--lora_ft", type=str, default=None, help="Path to LoRA adapter for FT (optional)")
    p.add_argument("--merge_lora", action="store_true", help="Merge LoRA weights into the model")
    p.add_argument("--layer_index", type=int, default=6, help="Transformer layer index to hook (0-based)")
    p.add_argument("--max_prompts", type=int, default=None, help="Max number of prompt groups")
    p.add_argument("--max_paraphrases", type=int, default=None, help="Max paraphrases per group")
    p.add_argument("--max_length", type=int, default=512, help="Max sequence length (tokens)")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for forward passes")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (default auto)")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto","bfloat16","float16","float32"], help="Torch dtype")
    p.add_argument("--topk_vocab", type=int, default=128, help="Top-K union vocab size for sensitive subspace ridge")
    p.add_argument("--topk_sv", type=int, default=8, help="Top singular vectors for sensitive subspace")
    p.add_argument("--ridge_lam", type=float, default=1e-3, help="Ridge regularization lambda")
    p.add_argument("--include_input_tag", action="store_true", help="Append '\\n\\nInput: {input}' when input non-empty")
    p.add_argument("--no_expA", action="store_true", help="Disable Experiment A (projection patch)")
    p.add_argument("--no_expB", action="store_true", help="Disable Experiment B (subspace swap)")
    p.add_argument("--no_mediation", action="store_true", help="Disable mediation analysis")
    return p

def main():
    ap = build_argparser()
    args = ap.parse_args()

    cfg = RunConfig(
        jsonl_path=args.jsonl,
        outdir=args.outdir,
        model_base=args.model_base,
        model_ft=args.model_ft,
        lora_ft=args.lora_ft,
        merge_lora=bool(args.merge_lora),
        layer_index=int(args.layer_index),
        max_prompts=args.max_prompts,
        max_paraphrases=args.max_paraphrases,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        topk_vocab=args.topk_vocab,
        topk_sv=args.topk_sv,
        ridge_lam=args.ridge_lam,
        include_input_tag=bool(args.include_input_tag),
        do_experiment_A=(not bool(args.no_expA)),
        do_experiment_B=(not bool(args.no_expB)),
        do_mediation=(not bool(args.no_mediation)),
    )

    runner = RobustInvarianceRunner(cfg)
    runner.run()

if __name__ == "__main__":
    main()
