#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import math
import time
import copy
import random
import logging
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional PEFT for LoRA
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Try reusing helpers from uploaded code if available
EXTERNAL_HELPERS = {}
for helper_path in ["/mnt/data/file1.py", "/mnt/data/file2.py"]:
    if os.path.exists(helper_path):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(os.path.basename(helper_path).replace(".py",""), helper_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            EXTERNAL_HELPERS.update({name: getattr(mod, name) for name in dir(mod)})
        except Exception as e:
            print(f"[WARN] Could not import {helper_path}: {e}", file=sys.stderr)

# Utilities

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return 1.96 * (x.std(ddof=1) / math.sqrt(max(1, len(x))))

def softmax_logits(logits: torch.Tensor, dim: int=-1) -> torch.Tensor:
    return F.softmax(logits, dim=dim)

def sym_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (F.kl_div(q.log(), p, reduction="none").sum(-1) + F.kl_div(p.log(), q, reduction="none").sum(-1))

# Config / CLI

@dataclass
class Config:
    base_model: str
    tokenizer_name: Optional[str]
    ft_model: Optional[str]
    lora_adapter: Optional[str]
    merge_lora: bool
    data_json: str
    output_dir: str
    device: str
    dtype: str
    batch_size: int
    max_families: int
    max_paraphrases: int
    layers: str
    heads_to_test: str
    topk_vocab: int
    subspace_k: int
    jacobian_mode: str
    jacobian_eps: float
    seed: int
    run_H1: bool
    run_H2: bool
    run_H3: bool
    run_H4: bool
    run_H5: bool
    run_H6: bool

def parse_args():
    parser = argparse.ArgumentParser(description="Paraphrase-robustness mechanism experiments")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--ft-model", type=str, default=None)
    parser.add_argument("--lora-adapter", type=str, default=None)
    parser.add_argument("--merge-lora", action="store_true")
    parser.add_argument("--data-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-families", type=int, default=100)
    parser.add_argument("--max-paraphrases", type=int, default=16)
    parser.add_argument("--layers", type=str, default="6")
    parser.add_argument("--heads-to-test", type=str, default="4,7")
    parser.add_argument("--topk-vocab", type=int, default=128)
    parser.add_argument("--subspace-k", type=int, default=8)
    parser.add_argument("--jacobian-mode", type=str, default="jacobian", choices=["jacobian","ridge"])
    parser.add_argument("--jacobian-eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-H1", action="store_true")
    parser.add_argument("--run-H2", action="store_true")
    parser.add_argument("--run-H3", action="store_true")
    parser.add_argument("--run-H4", action="store_true")
    parser.add_argument("--run-H5", action="store_true")
    parser.add_argument("--run-H6", action="store_true")
    args = parser.parse_args()

    # Let external helper override if present
    if "parse_args" in EXTERNAL_HELPERS:
        try:
            ext_args = EXTERNAL_HELPERS["parse_args"]()
            for k, v in vars(ext_args).items():
                if hasattr(args, k) and v is not None:
                    setattr(args, k, v)
        except Exception as e:
            print(f"[WARN] external parse_args failed: {e}", file=sys.stderr)

    return Config(
        base_model=args.base_model,
        tokenizer_name=args.tokenizer_name,
        ft_model=args.ft_model,
        lora_adapter=args.lora_adapter,
        merge_lora=args.merge_lora,
        data_json=args.data_json,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_families=args.max_families,
        max_paraphrases=args.max_paraphrases,
        layers=args.layers,
        heads_to_test=args.heads_to_test,
        topk_vocab=args.topk_vocab,
        subspace_k=args.subspace_k,
        jacobian_mode=args.jacobian_mode,
        jacobian_eps=args.jacobian_eps,
        seed=args.seed,
        run_H1=args.run_H1,
        run_H2=args.run_H2,
        run_H3=args.run_H3,
        run_H4=args.run_H4,
        run_H5=args.run_H5,
        run_H6=args.run_H6,
    )

# Logging

def setup_logging(outdir: str):
    ensure_dir(outdir)
    log_path = os.path.join(outdir, "run.log")
    logger = logging.getLogger("robust")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

# Data loader

def load_prompts(path: str) -> List[dict]:
    """
    Load either JSON (.json) or JSON Lines (.jsonl).
    For .jsonl, skip blank/commented lines and log decode errors rather than crash.
    """
    if path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] JSONL decode error at line {ln}: {e}", file=sys.stderr)
                    continue
        return items
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt_families(items: List[dict],
                          max_families: int,
                          max_paraphrases: int,
                          logger: logging.Logger) -> List[dict]:
    """
    Build families robustly.
    """
    if not items:
        logger.warning("No items loaded from data file.")
        return []

    families: List[dict] = []

    def attach_input(instr: str, inp: str) -> str:
        instr = (instr or "").strip()
        inp = (inp or "").strip()
        if inp:
            return f"{instr}\n\nInput: {inp}"
        return instr

    for obj in items:
        if not isinstance(obj, dict):
            continue

        instr_orig = obj.get("instruction_original") or obj.get("instruction") \
                     or obj.get("prompt") or obj.get("question") or obj.get("text")
        inpt = obj.get("input") or obj.get("context") or ""
        paras_field = obj.get("paraphrases")
        paraphrases: List[Tuple[str, str]] = []

        if isinstance(paras_field, list):
            for i, it in enumerate(paras_field):
                if isinstance(it, dict):
                    txt = it.get("text")
                    key = it.get("key", f"paraphrases_{i}")
                    if isinstance(txt, str) and txt.strip():
                        paraphrases.append((str(key), txt.strip()))
                elif isinstance(it, str) and it.strip():
                    paraphrases.append((f"paraphrases_{i}", it.strip()))

        if not paraphrases:
            for k, v in obj.items():
                if isinstance(v, str) and k.startswith(("instruct_", "paraphrase_", "variant_", "rephrase_", "aug_", "alt_")) and v.strip():
                    paraphrases.append((k, v.strip()))

        if isinstance(instr_orig, str) and instr_orig.strip() and paraphrases:
            orig_txt = attach_input(instr_orig, inpt)
            paraphrases = [(k, attach_input(t, inpt)) for k, t in paraphrases[:max_paraphrases]]
            families.append({
                "original": orig_txt,
                "paraphrases": paraphrases,
                "prompt_count": obj.get("prompt_count"),
            })
            if len(families) >= max_families:
                break

    if not families:
        logger.warning("Could not construct any families from dataset.")

    for i, fam in enumerate(families):
        fam["index"] = i
    logger.info(f"Built {len(families)} prompt families (each with up to {max_paraphrases} paraphrases).")
    return families

# Model loading

DTYPE_MAP = {
    "auto": None,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

class ModelBundle:
    def __init__(self, model, tokenizer, device, dtype_str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype_str = dtype_str

    @staticmethod
    def load(config: Config, which: str, logger: logging.Logger, merge_lora_override: Optional[bool]=None):
        tok_name = config.tokenizer_name or config.base_model
        tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
        dtype = DTYPE_MAP.get(config.dtype, None)
        torch_dtype = dtype
        device = config.device

        if which == "BASE":
            model_id = config.base_model
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, device_map=None
            )
            model.to(device)
            model.eval()
            return ModelBundle(model, tokenizer, device, config.dtype)

        if config.ft_model:
            logger.info(f"Loading FT as fully-finetuned model: {config.ft_model}")
            model = AutoModelForCausalLM.from_pretrained(config.ft_model, torch_dtype=torch_dtype, device_map=None)
            model.to(device); model.eval()
            return ModelBundle(model, tokenizer, device, config.dtype)

        if config.lora_adapter:
            if not PEFT_AVAILABLE:
                raise RuntimeError("peft not installed, cannot load LoRA adapter")
            logger.info(f"Loading base + LoRA adapter: {config.base_model} + {config.lora_adapter}")
            base = AutoModelForCausalLM.from_pretrained(config.base_model, torch_dtype=torch_dtype, device_map=None)
            base.to(device); base.eval()
            peft_model = PeftModel.from_pretrained(base, config.lora_adapter)
            if config.merge_lora if merge_lora_override is None else merge_lora_override:
                logger.info("Merging LoRA weights into base (merge_and_unload)")
                peft_model = peft_model.merge_and_unload()
            return ModelBundle(peft_model, tokenizer, device, config.dtype)

        raise ValueError("To load FT, provide --ft-model or --lora-adapter")

# Tokenization & batching

def tokenize_batch(tokenizer, texts: List[str], device: str, add_bos: bool=True):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    lengths = attn.sum(-1)
    last_ix = (lengths - 1).clamp_min(0)
    return {"input_ids": input_ids, "attention_mask": attn, "last_index": last_ix}

# Layer/station capture & patching

class LayerStations:
    """Capture stations inside a decoder block's MLP and provide patch points."""
    def __init__(self, model, layer_idx: int, logger: logging.Logger):
        self.model = model
        self.layer_idx = layer_idx
        self.logger = logger

        # Resolve modules (LLaMA/Gemma-like)
        try:
            self.block = model.model.layers[layer_idx]
        except Exception:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                self.block = model.model.layers[layer_idx]
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                self.block = model.transformer.h[layer_idx]
            else:
                raise RuntimeError("Cannot find layer module; please adapt for your architecture.")

        self.attn = getattr(self.block, "self_attn", None)
        self.mlp = getattr(self.block, "mlp", None)
        if self.mlp is None:
            raise RuntimeError("Layer has no 'mlp' attribute—adjust code for your model.")

        # Common submodules
        self.gate_proj = getattr(self.mlp, "gate_proj", None)
        self.up_proj = getattr(self.mlp, "up_proj", None)
        self.down_proj = getattr(self.mlp, "down_proj", None)

        # Activation resolution: match model's own (e.g., 'gelu_pytorch_tanh')
        self.hidden_act = getattr(self.model.config, "hidden_act", "gelu")
        self.act_fn = self._resolve_act_fn(self.hidden_act)

        # Buffers & patch state
        self.captures: Dict[str, torch.Tensor] = {}
        self.patch_down_last: Optional[Callable] = None
        self.ablate_heads: List[int] = []
        self._hooks = []
        self._register_hooks()

    def _resolve_act_fn(self, name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        n = (name or "").lower()
        if n in ("silu", "swiglu"):
            return F.silu
        if n.startswith("gelu"):  # covers gelu_pytorch_tanh, gelu_fast, etc.
            return lambda x: F.gelu(x, approximate="tanh")
        # Sensible default
        self.logger.warning(f"Unrecognized hidden_act '{name}', using PyTorch GELU(tanh).")
        return lambda x: F.gelu(x, approximate="tanh")

    # Robust reset + legacy alias
    def reset_captures(self):
        self.captures = {}

    def clear(self):
        self.reset_captures()

    def remove(self):
        for h in self._hooks:
            try: h.remove()
            except: pass
        self._hooks = []

    def _register_hooks(self):
        # RES: input to MLP
        def mlp_pre_hook(module, inputs):
            x = inputs[0].detach()
            self.captures["RES"] = x
        self._hooks.append(self.mlp.register_forward_pre_hook(mlp_pre_hook))

        # Gate pre & act
        if self.gate_proj is not None:
            def gate_hook(module, inputs, output):
                out = output.detach()
                self.captures["GATE_PRE"] = out
                try:
                    self.captures["GATE_ACT"] = self.act_fn(out)
                except Exception:
                    self.captures["GATE_ACT"] = out
            self._hooks.append(self.gate_proj.register_forward_hook(gate_hook))

        # UP
        if self.up_proj is not None:
            def up_hook(module, inputs, output):
                self.captures["UP"] = output.detach()
            self._hooks.append(self.up_proj.register_forward_hook(up_hook))

        # DOWN 
        def mlp_hook(module, inputs, output):
            out = output
            if self.patch_down_last is not None and isinstance(out, torch.Tensor):
                try:
                    b_ix = self.captures.get("_batch_indices", None)
                    l_ix = self.captures.get("_last_indices", None)
                    if b_ix is not None and l_ix is not None:
                        out = self.patch_down_last(out, b_ix, l_ix)
                except Exception as e:
                    self.logger.error(f"Patch failed: {e}")
            self.captures["DOWN"] = out.detach()
            return out
        self._hooks.append(self.mlp.register_forward_hook(mlp_hook))

        # Attention head ablation
        if self.attn is not None:
            def attn_out_hook(module, inputs, output):
                if not self.ablate_heads:
                    return output

                # Handle tuple vs tensor returns
                if isinstance(output, tuple):
                    hs = output[0]
                    others = output[1:]
                else:
                    hs = output
                    others = None

                if not torch.is_tensor(hs):
                    return output

                D = hs.shape[-1]
                num_heads = getattr(module, "num_heads", None) or getattr(module, "n_heads", None)
                if not num_heads or D % int(num_heads) != 0:
                    return output

                head_dim = D // int(num_heads)
                out_hs = hs.clone()
                for h in self.ablate_heads:
                    if 0 <= h < int(num_heads):
                        s = h * head_dim
                        e = (h + 1) * head_dim
                        out_hs[..., s:e] = 0.0

                return (out_hs, *others) if others is not None else out_hs

            self._hooks.append(self.attn.register_forward_hook(attn_out_hook))

    def set_down_patch(self, patch_callable: Optional[Callable]):
        self.patch_down_last = patch_callable

    def set_head_ablation(self, heads: List[int]):
        self.ablate_heads = list(heads or [])

    def forward_with_capture(self, mb: ModelBundle, texts: List[str], batch_size: int,
                             last_only: bool=True) -> Dict[str, torch.Tensor]:
        model = mb.model
        tok = mb.tokenizer
        device = mb.device

        self.reset_captures()
        outs = {k: [] for k in ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN","LOGITS","LAST_IX"]}

        # Determine hidden size D safely
        D = int(getattr(model.config, "hidden_size", None) or
                getattr(getattr(self.block, "self_attn", object()), "hidden_size", None) or 0)
        if D <= 0:
            raise RuntimeError("Cannot infer hidden size for captures.")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenize_batch(tok, batch_texts, device)
            input_ids = enc["input_ids"]; attn = enc["attention_mask"]; last_ix = enc["last_index"]

            # for patch function (if used)
            self.captures["_batch_indices"] = torch.arange(input_ids.size(0), device=device, dtype=torch.long)
            self.captures["_last_indices"] = last_ix

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=False)
                logits = out.logits  # [B,T,V]

            B, T = input_ids.size(0), input_ids.size(1)

            def zeros_BT_D():
                return logits.new_zeros((B, T, D))

            # POST = UP * act(GATE_PRE) if both exist
            gate_pre = self.captures.get("GATE_PRE", None)
            up = self.captures.get("UP", None)
            if gate_pre is not None and up is not None:
                try:
                    post = up * self.act_fn(gate_pre)
                except Exception:
                    post = zeros_BT_D()
            else:
                post = zeros_BT_D()

            # Collect with safe placeholders
            for name in ["RES","UP","GATE_PRE","GATE_ACT","DOWN"]:
                tens = self.captures.get(name, None)
                if tens is None:
                    tens = zeros_BT_D()
                outs[name].append(tens)
            outs["POST"].append(post)
            outs["LOGITS"].append(logits.detach())
            outs["LAST_IX"].append(last_ix.detach())

        # Concatenate batches
        for k in ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN","LOGITS","LAST_IX"]:
            outs[k] = torch.cat(outs[k], dim=0)

        # Pool to last token if requested
        if last_only:
            pooled = {}
            last_ix = outs["LAST_IX"]
            _, T, Dp = outs["DOWN"].shape
            sel = last_ix.view(-1,1,1).expand(-1,1,Dp)
            for k in ["RES","UP","GATE_PRE","GATE_ACT","POST","DOWN"]:
                pooled[k] = outs[k].gather(1, sel).squeeze(1).contiguous()  # [N,D]
            pooled["LOGITS"] = outs["LOGITS"].gather(1, last_ix.view(-1,1,1).expand(-1,1,outs["LOGITS"].size(-1))).squeeze(1)
            pooled["LAST_IX"] = last_ix
            return pooled
        else:
            return outs

# Subspace (Jacobian power iteration / ridge)

def select_vocab_from_logits(logits: torch.Tensor, tokenizer, topk: int) -> torch.Tensor:
    if logits.dim() == 2:
        logits = logits[0]
    topk_ix = torch.topk(logits, k=min(topk, logits.numel()), dim=-1).indices
    return topk_ix

class SubspaceEstimator:
    def __init__(self, bundle: ModelBundle, stations: LayerStations, layer_idx: int,
                 tokenizer, logger: logging.Logger, jacobian_eps=1e-3):
        self.mb = bundle
        self.sta = stations
        self.layer_idx = layer_idx
        self.tok = tokenizer
        self.logger = logger
        self.eps = jacobian_eps

    def _forward_with_down_override(self, h_new_delta: torch.Tensor,
                                    base_inputs, vocab_ix: torch.Tensor) -> torch.Tensor:
        model = self.mb.model
        device = self.mb.device

        input_ids = base_inputs["input_ids"]
        attn = base_inputs["attention_mask"]
        last_ix = base_inputs["last_index"]
        B = input_ids.size(0)
        assert B == 1, "This function expects B=1 (original text only)."

        def patch_fn(out, b_ix, l_ix):
            out = out.clone()
            t0 = int(l_ix[0].item())
            T = out.size(1)
            if t0 >= T:
                t0 = T - 1
            base_vec = self._down_orig  # [D]
            new_vec = base_vec + h_new_delta  # [D]
            out[0, t0, :] = new_vec
            return out

        self.sta.set_down_patch(patch_fn)
        self.sta.captures["_batch_indices"] = torch.arange(B, device=device, dtype=torch.long)
        self.sta.captures["_last_indices"] = last_ix

        with torch.enable_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits  # [B,T,V]
            last_logits = logits[0, last_ix[0], :]  # [V]
            sel = last_logits.index_select(0, vocab_ix)  # [m]
        self.sta.set_down_patch(None)
        return sel

    def compute_projector(self, orig_inputs: dict, h_down_orig: torch.Tensor,
                          vocab_ix: torch.Tensor, k: int, mode: str="jacobian") -> torch.Tensor:
        device = self.mb.device
        d = h_down_orig.numel()
        self._down_orig = h_down_orig.detach()
        dtype_h = h_down_orig.dtype  # likely float16/bfloat16

        if mode == "ridge":
            V = torch.randn(d, k, device=device, dtype=torch.float32)
            V, _ = torch.linalg.qr(V, mode="reduced")
            return V @ V.T

        def f(delta_vec: torch.Tensor) -> torch.Tensor:
            return self._forward_with_down_override(delta_vec, orig_inputs, vocab_ix)

        def Jv(v: torch.Tensor) -> torch.Tensor:
            v = v.to(dtype_h).detach().requires_grad_(False)
            with torch.enable_grad():
                delta0 = torch.zeros_like(h_down_orig, requires_grad=True)
                _, jvp = torch.autograd.functional.jvp(func=f, inputs=(delta0,), v=(v,), create_graph=False, strict=True)
            return jvp.detach()

        def JT_w(w: torch.Tensor) -> torch.Tensor:
            w = w.to(dtype_h).detach()
            with torch.enable_grad():
                delta0 = torch.zeros_like(h_down_orig, requires_grad=True)
                y = f(delta0)
                scalar = (y * w).sum()
                grad = torch.autograd.grad(scalar, delta0, retain_graph=False, create_graph=False)[0]
            return grad.detach()

        V = torch.randn(d, k, device=device, dtype=torch.float32)
        V, _ = torch.linalg.qr(V, mode="reduced")  # float32 QR
        iters = 12
        for _ in range(iters):
            JV_cols = [Jv(V[:, j]) for j in range(k)]
            JV = torch.stack(JV_cols, dim=1)
            Y_cols = [JT_w(JV[:, j]) for j in range(k)]
            Y = torch.stack(Y_cols, dim=1)
            V, _ = torch.linalg.qr(Y.float(), mode="reduced")  # float32 QR
        P = V @ V.T
        return P

# Head transplant helper (H3)

def transplant_heads_LLaMA_like(dst_block, src_block, head_indices: List[int], logger: logging.Logger):
    attn_dst = getattr(dst_block, "self_attn", None)
    attn_src = getattr(src_block, "self_attn", None)
    if attn_dst is None or attn_src is None:
        logger.warning("Cannot find self_attn for transplant; skipping.")
        return False

    qd, kd, vd, od = attn_dst.q_proj, attn_dst.k_proj, attn_dst.v_proj, attn_dst.o_proj
    qs, ks, vs, osrc = attn_src.q_proj, attn_src.k_proj, attn_src.v_proj, attn_src.o_proj
    if any(m is None for m in [qd,kd,vd,od,qs,ks,vs,osrc]):
        logger.warning("Missing proj modules; skipping.")
        return False

    num_heads = getattr(attn_dst, "num_heads", None) or getattr(attn_dst, "n_heads", None)
    embed_dim = qd.out_features
    head_dim = embed_dim // int(num_heads)

    with torch.no_grad():
        for h in head_indices:
            s, e = h*head_dim, (h+1)*head_dim
            qd.weight[s:e, :] = qs.weight[s:e, :]
            kd.weight[s:e, :] = ks.weight[s:e, :]
            vd.weight[s:e, :] = vs.weight[s:e, :]
            if qd.bias is not None and qs.bias is not None:
                qd.bias[s:e] = qs.bias[s:e]
                kd.bias[s:e] = ks.bias[s:e]
                vd.bias[s:e] = vs.bias[s:e]
            od.weight[:, s:e] = osrc.weight[:, s:e]
    return True

# Plotting helper

def bar_with_ci(ax, labels, means, cis, title, ylabel):
    x = np.arange(len(labels))
    ax.bar(x, means)
    ax.errorbar(x, means, yerr=cis, fmt='none', ecolor='black', capsize=4, linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)

# Main Experiment Runner

class Runner:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.layers = [int(x.strip()) for x in cfg.layers.split(",") if x.strip()]
        self.heads = [int(x.strip()) for x in cfg.heads_to_test.split(",") if x.strip()]
        self.out = cfg.output_dir
        ensure_dir(self.out)

    def load_models(self):
        self.logger.info("Loading BASE model...")
        if "load_model" in EXTERNAL_HELPERS:
            try:
                self.base = EXTERNAL_HELPERS["load_model"](self.cfg.base_model)
                if isinstance(self.base, tuple):
                    m, tok = self.base
                    self.base = ModelBundle(m, tok, self.cfg.device, self.cfg.dtype)
            except Exception as e:
                self.logger.warning(f"external load_model failed; falling back. Error: {e}")
                self.base = ModelBundle.load(self.cfg, "BASE", self.logger)
        else:
            self.base = ModelBundle.load(self.cfg, "BASE", self.logger)

        self.logger.info("Loading FT model...")
        self.ft = ModelBundle.load(self.cfg, "FT", self.logger)

    def load_data(self):
        items = load_prompts(self.cfg.data_json)
        self.families = build_prompt_families(items, self.cfg.max_families, self.cfg.max_paraphrases, self.logger)
        for i, fam in enumerate(self.families):
            fam["index"] = i
        self.logger.info(f"Loaded {len(self.families)} prompt families.")

    # H1
    def run_H1(self):
        self.logger.info("=== H1: Orientation change at DOWN (Δ f_sens) ===")
        outdir = os.path.join(self.out, "H1_orientation")
        ensure_dir(outdir)

        for layer_idx in self.layers:
            self.logger.info(f"[H1] Layer {layer_idx}")
            rows_layer = []
            sta_base = LayerStations(self.base.model, layer_idx, self.logger)
            sta_ft   = LayerStations(self.ft.model,   layer_idx, self.logger)

            for fam in tqdm(self.families, desc=f"H1 L{layer_idx} families", ncols=100):
                orig = fam["original"]
                paras = [t for _, t in fam["paraphrases"]]

                # BASE
                base_orig_caps = sta_base.forward_with_capture(self.base, [orig], self.cfg.batch_size, last_only=True)
                base_pars_caps = sta_base.forward_with_capture(self.base, paras, self.cfg.batch_size, last_only=True)
                base_orig_down = base_orig_caps["DOWN"][0]
                base_pars_down = base_pars_caps["DOWN"]
                vocab_ix = select_vocab_from_logits(base_orig_caps["LOGITS"][0], self.base.tokenizer, self.cfg.topk_vocab)
                se_base = SubspaceEstimator(self.base, sta_base, layer_idx, self.base.tokenizer, self.logger, jacobian_eps=self.cfg.jacobian_eps)
                P_base = se_base.compute_projector(
                    orig_inputs=tokenize_batch(self.base.tokenizer, [orig], self.base.device),
                    h_down_orig=base_orig_down, vocab_ix=vocab_ix,
                    k=self.cfg.subspace_k, mode=self.cfg.jacobian_mode
                )  # float32

                base_fs = []
                for i in range(base_pars_down.size(0)):
                    delta = (base_pars_down[i] - base_orig_down).detach().float()
                    num = (delta @ (P_base @ delta)).item()
                    den = (delta @ delta).item() + 1e-9
                    base_fs.append(num/den)

                # FT
                ft_orig_caps = sta_ft.forward_with_capture(self.ft, [orig], self.cfg.batch_size, last_only=True)
                ft_pars_caps = sta_ft.forward_with_capture(self.ft, paras, self.cfg.batch_size, last_only=True)
                ft_orig_down = ft_orig_caps["DOWN"][0]
                ft_pars_down = ft_pars_caps["DOWN"]
                vocab_ix_ft = select_vocab_from_logits(ft_orig_caps["LOGITS"][0], self.ft.tokenizer, self.cfg.topk_vocab)
                se_ft = SubspaceEstimator(self.ft, sta_ft, layer_idx, self.ft.tokenizer, self.logger, jacobian_eps=self.cfg.jacobian_eps)
                P_ft = se_ft.compute_projector(
                    orig_inputs=tokenize_batch(self.ft.tokenizer, [orig], self.ft.device),
                    h_down_orig=ft_orig_down, vocab_ix=vocab_ix_ft,
                    k=self.cfg.subspace_k, mode=self.cfg.jacobian_mode
                )  # float32

                ft_fs = []
                for i in range(ft_pars_down.size(0)):
                    delta = (ft_pars_down[i] - ft_orig_down).detach().float()
                    num = (delta @ (P_ft @ delta)).item()
                    den = (delta @ delta).item() + 1e-9
                    ft_fs.append(num/den)

                rows_layer.append({
                    "family_index": fam["index"],
                    "layer": layer_idx,
                    "base_f_sens_mean": float(np.mean(base_fs)),
                    "ft_f_sens_mean": float(np.mean(ft_fs)),
                    "delta_f_sens": float(np.mean(ft_fs) - np.mean(base_fs)),
                    "n_paraphrases": int(len(paras)),
                })

            df = pd.DataFrame(rows_layer)
            csv_path = os.path.join(outdir, f"H1_L{layer_idx}_rows.csv")
            df.to_csv(csv_path, index=False)
            self.logger.info(f"[H1] Saved {csv_path}")

            if df.empty:
                self.logger.warning(f"[H1] No rows for layer {layer_idx}.")
                continue

            df_layer = df[df["layer"] == layer_idx]
            means = [df_layer["base_f_sens_mean"].mean(), df_layer["ft_f_sens_mean"].mean()]
            cis = [ci95(df_layer["base_f_sens_mean"].values), ci95(df_layer["ft_f_sens_mean"].values)]
            fig, ax = plt.subplots(figsize=(6,4))
            bar_with_ci(ax, ["BASE","FT"], means, cis, f"H1 L{layer_idx}: mean f_sens (DOWN)", "f_sens")
            fig.tight_layout(); fig.savefig(os.path.join(outdir, f"H1_L{layer_idx}_f_sens_bar.png"), dpi=160); plt.close(fig)

            fig, ax = plt.subplots(figsize=(6,4))
            ax.hist(df_layer["delta_f_sens"].values, bins=30); ax.set_title(f"H1 L{layer_idx}: Δ f_sens = FT-BASE")
            ax.grid(True, linestyle=":", alpha=0.5)
            fig.tight_layout(); fig.savefig(os.path.join(outdir, f"H1_L{layer_idx}_delta_hist.png"), dpi=160); plt.close(fig)

    # H2
    def run_H2(self):
        self.logger.info("=== H2: Projection interventions (sens-only vs orth-only) ===")
        outdir = os.path.join(self.out, "H2_projection")
        ensure_dir(outdir)

        for layer_idx in self.layers:
            self.logger.info(f"[H2] Layer {layer_idx}")
            rows_layer = []
            sta_ft = LayerStations(self.ft.model, layer_idx, self.logger)

            for fam in tqdm(self.families, desc=f"H2 L{layer_idx} families", ncols=100):
                orig = fam["original"]
                paras = [t for _, t in fam["paraphrases"]]

                # FT captures
                ft_orig_caps = sta_ft.forward_with_capture(self.ft, [orig], self.cfg.batch_size, last_only=True)
                ft_pars_caps = sta_ft.forward_with_capture(self.ft, paras, self.cfg.batch_size, last_only=True)
                ft_orig_down = ft_orig_caps["DOWN"][0]
                ft_pars_down = ft_pars_caps["DOWN"]

                # projector from original logits (float32)
                vocab_ix = select_vocab_from_logits(ft_orig_caps["LOGITS"][0], self.ft.tokenizer, self.cfg.topk_vocab)
                se_ft = SubspaceEstimator(self.ft, sta_ft, layer_idx, self.ft.tokenizer, self.logger, jacobian_eps=self.cfg.jacobian_eps)
                P = se_ft.compute_projector(
                    orig_inputs=tokenize_batch(self.ft.tokenizer, [orig], self.ft.device),
                    h_down_orig=ft_orig_down, vocab_ix=vocab_ix,
                    k=self.cfg.subspace_k, mode=self.cfg.jacobian_mode
                )

                p = softmax_logits(ft_orig_caps["LOGITS"][0]).detach()

                for i, para in enumerate(paras):
                    q = softmax_logits(ft_pars_caps["LOGITS"][i]).detach()
                    base_skl = sym_kl(p, q).item()

                    # project in float32
                    delta32 = (ft_pars_down[i] - ft_orig_down).detach().float()
                    d_sens32 = (P @ delta32)
                    d_orth32 = (delta32 - d_sens32)

                    # rescale to ||delta||
                    target = delta32.norm() + 1e-9
                    if d_sens32.norm() > 0:
                        d_sens32 = d_sens32 * (target / d_sens32.norm())
                    if d_orth32.norm() > 0:
                        d_orth32 = d_orth32 * (target / d_orth32.norm())

                    def make_patch(vec32):
                        base_vec = ft_orig_down  # dtype from model
                        vec = vec32.to(base_vec.dtype)
                        def patch_fn(out, b_ix, l_ix):
                            out = out.clone()
                            t0 = int(l_ix[0].item())
                            T = out.size(1)
                            if t0 >= T:
                                t0 = T - 1  # guard against stale/mismatched index
                            out[0, t0, :] = base_vec + vec
                            return out
                        return patch_fn

                    enc_para = tokenize_batch(self.ft.tokenizer, [para], self.cfg.device)

                    # sens-only (reset captures + set correct indices)
                    sta_ft.reset_captures()
                    sta_ft.set_down_patch(make_patch(d_sens32))
                    sta_ft.captures["_batch_indices"] = torch.arange(enc_para["input_ids"].size(0),
                                                                     device=self.cfg.device, dtype=torch.long)
                    sta_ft.captures["_last_indices"] = enc_para["last_index"]
                    with torch.no_grad():
                        out_sens = self.ft.model(
                            input_ids=enc_para["input_ids"],
                            attention_mask=enc_para["attention_mask"]
                        )
                        logits_sens = out_sens.logits[0, enc_para["last_index"][0], :]
                    sta_ft.set_down_patch(None)

                    # orth-only
                    sta_ft.reset_captures()
                    sta_ft.set_down_patch(make_patch(d_orth32))
                    sta_ft.captures["_batch_indices"] = torch.arange(enc_para["input_ids"].size(0),
                                                                     device=self.cfg.device, dtype=torch.long)
                    sta_ft.captures["_last_indices"] = enc_para["last_index"]
                    with torch.no_grad():
                        out_orth = self.ft.model(
                            input_ids=enc_para["input_ids"],
                            attention_mask=enc_para["attention_mask"]
                        )
                        logits_orth = out_orth.logits[0, enc_para["last_index"][0], :]
                    sta_ft.set_down_patch(None)

                    q_sens = softmax_logits(logits_sens).detach()
                    q_orth = softmax_logits(logits_orth).detach()
                    skl_sens = sym_kl(p, q_sens).item()
                    skl_orth = sym_kl(p, q_orth).item()

                    rows_layer.append({
                        "family_index": fam["index"],
                        "layer": layer_idx,
                        "para_idx": i,
                        "sKL_FT": base_skl,
                        "sKL_sens_only": skl_sens,
                        "sKL_orth_only": skl_orth,
                        "delta_sens_minus_FT": skl_sens - base_skl,
                        "delta_orth_minus_FT": skl_orth - base_skl,
                    })

            df = pd.DataFrame(rows_layer)
            csv_path = os.path.join(outdir, f"H2_L{layer_idx}_rows.csv")
            df.to_csv(csv_path, index=False)
            self.logger.info(f"[H2] Saved {csv_path}")

            if df.empty:
                self.logger.warning(f"[H2] No rows for layer {layer_idx}.")
                continue

            d1 = df["delta_sens_minus_FT"].values
            d2 = df["delta_orth_minus_FT"].values
            fig, ax = plt.subplots(figsize=(6,4))
            bar_with_ci(ax, ["sens-only − FT", "orth-only − FT"],
                        [np.mean(d1), np.mean(d2)],
                        [ci95(d1), ci95(d2)],
                        f"H2 L{layer_idx}: Projection interventions", "Δ sKL")
            fig.tight_layout(); fig.savefig(os.path.join(outdir, f"H2_L{layer_idx}_bars.png"), dpi=160); plt.close(fig)

            fig, ax = plt.subplots(1,2, figsize=(10,4))
            ax[0].hist(d1, bins=30); ax[0].set_title("sens-only − FT")
            ax[1].hist(d2, bins=30); ax[1].set_title("orth-only − FT")
            for a in ax: a.grid(True, linestyle=":", alpha=0.5)
            fig.tight_layout(); fig.savefig(os.path.join(outdir, f"H2_L{layer_idx}_hists.png"), dpi=160); plt.close(fig)

    # H3
    def run_H3(self):
        self.logger.info("=== H3: Head necessity (ablations) & sufficiency (transplant) ===")
        outdir = os.path.join(self.out, "H3_heads")
        ensure_dir(outdir)

        rows_abl = []
        rows_trans = []

        for layer_idx in self.layers:
            self.logger.info(f"[H3] Layer {layer_idx}")
            heads = self.heads
            for head in heads + [-1]:
                sta_ft = LayerStations(self.ft.model, layer_idx, self.logger)
                if head >= 0:
                    sta_ft.set_head_ablation([head])
                    tag = f"head{head}"
                else:
                    sta_ft.set_head_ablation([])
                    tag = "control"

                for fam in tqdm(self.families, desc=f"H3 Abl L{layer_idx} {tag}", ncols=100):
                    orig = fam["original"]
                    paras = [t for _, t in fam["paraphrases"]]
                    caps = sta_ft.forward_with_capture(self.ft, [orig] + paras, self.cfg.batch_size, last_only=True)
                    p = softmax_logits(caps["LOGITS"][0]).detach()
                    for i in range(1, caps["LOGITS"].size(0)):
                        q = softmax_logits(caps["LOGITS"][i]).detach()
                        skl = sym_kl(p, q).item()
                        rows_abl.append({
                            "family_index": fam["index"],
                            "layer": layer_idx,
                            "head": head,
                            "sKL": skl,
                        })

            base_clone = copy.deepcopy(self.base.model).eval().to(self.cfg.device)
            dst_block = base_clone.model.layers[layer_idx]
            src_block = self.ft.model.model.layers[layer_idx]
            ok = transplant_heads_LLaMA_like(dst_block, src_block, self.heads, self.logger)
            if ok:
                tok = self.base.tokenizer
                for fam in tqdm(self.families, desc=f"H3 Transplant L{layer_idx}", ncols=100):
                    orig = fam["original"]
                    paras = [t for _, t in fam["paraphrases"]]
                    enc = tokenize_batch(tok, [orig]+paras, self.cfg.device)
                    with torch.no_grad():
                        out = base_clone(
                            input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"]
                        )
                    logits = out.logits
                    last_ix = enc["last_index"]
                    p = softmax_logits(logits[0, last_ix[0]])
                    for i in range(1, logits.size(0)):
                        q = softmax_logits(logits[i, last_ix[i]])
                        skl = sym_kl(p, q).item()
                        rows_trans.append({
                            "family_index": fam["index"],
                            "layer": layer_idx,
                            "transplant": "heads_"+",".join(map(str,self.heads)),
                            "sKL": skl,
                        })

        if rows_abl:
            df = pd.DataFrame(rows_abl)
            df.to_csv(os.path.join(outdir, "H3_ablation_rows.csv"), index=False)
            fig, ax = plt.subplots(figsize=(7,4))
            labels, means, cis = [], [], []
            for head_val in sorted(set(df["head"].tolist())):
                labels.append(f"h{head_val}" if head_val>=0 else "control")
                vals = df[df["head"]==head_val]["sKL"].values
                means.append(np.mean(vals))
                cis.append(ci95(vals))
            bar_with_ci(ax, labels, means, cis, "H3 Ablation: sKL per condition", "sKL")
            fig.tight_layout(); fig.savefig(os.path.join(outdir, "H3_ablation_bars.png"), dpi=160); plt.close(fig)

        if rows_trans:
            df = pd.DataFrame(rows_trans)
            df.to_csv(os.path.join(outdir, "H3_transplant_rows.csv"), index=False)
            fig, ax = plt.subplots(figsize=(6,4))
            vals = df["sKL"].values
            bar_with_ci(ax, ["BASE+transplanted"], [np.mean(vals)], [ci95(vals)],
                        "H3 Transplant: sKL", "sKL")
            fig.tight_layout(); fig.savefig(os.path.join(outdir, "H3_transplant_bar.png"), dpi=160); plt.close(fig)

    # H4
    def run_H4(self):
        self.logger.info("=== H4: DOWN weight interpolation dose–response ===")
        outdir = os.path.join(self.out, "H4_interp")
        ensure_dir(outdir)

        lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]  # 0=FT, 1=BASE
        rows = []

        for layer_idx in self.layers:
            self.logger.info(f"[H4] Layer {layer_idx}")
            ft_work = copy.deepcopy(self.ft.model).eval().to(self.cfg.device)

            down_ft = ft_work.model.layers[layer_idx].mlp.down_proj
            down_base = self.base.model.model.layers[layer_idx].mlp.down_proj
            W_ft = down_ft.weight.detach().clone()
            b_ft = down_ft.bias.detach().clone() if down_ft.bias is not None else None
            W_base = down_base.weight.detach().clone()
            b_base = down_base.bias.detach().clone() if down_base.bias is not None else None

            tok = self.ft.tokenizer

            for lam in lambdas:
                with torch.no_grad():
                    down_ft.weight[:] = (1-lam) * W_ft + lam * W_base
                    if b_ft is not None and b_base is not None:
                        down_ft.bias[:] = (1-lam) * b_ft + lam * b_base

                skls = []
                for fam in tqdm(self.families, desc=f"H4 L{layer_idx} λ={lam}", ncols=100):
                    orig = fam["original"]
                    paras = [t for _, t in fam["paraphrases"]]
                    enc = tokenize_batch(tok, [orig]+paras, self.cfg.device)
                    with torch.no_grad():
                        out = ft_work(
                            input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"]
                        )
                    logits = out.logits
                    last_ix = enc["last_index"]
                    p = softmax_logits(logits[0, last_ix[0]])
                    for i in range(1, logits.size(0)):
                        q = softmax_logits(logits[i, last_ix[i]])
                        skls.append(sym_kl(p, q).item())

                rows.append({
                    "layer": layer_idx,
                    "lambda": lam,
                    "mean_sKL": float(np.mean(skls) if skls else float("nan")),
                    "ci_sKL": float(ci95(np.array(skls))) if skls else float("nan"),
                })

            with torch.no_grad():
                down_ft.weight[:] = W_ft
                if b_ft is not None:
                    down_ft.bias[:] = b_ft

        df = pd.DataFrame(rows)
        csv_path = os.path.join(outdir, "H4_interp_rows.csv")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"[H4] Saved {csv_path}")

        for layer_idx in self.layers:
            sub = df[df["layer"] == layer_idx]
            if sub.empty:
                self.logger.warning(f"[H4] No rows for layer {layer_idx}.")
                continue
            fig, ax = plt.subplots(figsize=(6,4))
            ax.errorbar(sub["lambda"], sub["mean_sKL"], yerr=sub["ci_sKL"], marker="o")
            ax.set_title(f"H4 Layer {layer_idx}: sKL vs DOWN λ (0=FT, 1=BASE)")
            ax.set_xlabel("λ"); ax.set_ylabel("mean sKL")
            ax.grid(True, linestyle=":", alpha=0.5)
            fig.tight_layout(); fig.savefig(os.path.join(outdir, f"H4_L{layer_idx}_curve.png"), dpi=160); plt.close(fig)

    # H5
    def run_H5(self):
        self.logger.info("=== H5: Style-token vs content-token patches ===")
        outdir = os.path.join(self.out, "H5_style")
        ensure_dir(outdir)

        rows = []

        for layer_idx in self.layers:
            self.logger.info(f"[H5] Layer {layer_idx}")
            sta_ft = LayerStations(self.ft.model, layer_idx, self.logger)
            tok = self.ft.tokenizer

            for fam in tqdm(self.families, desc=f"H5 L{layer_idx} families", ncols=100):
                orig = fam["original"]
                paras = [t for _, t in fam["paraphrases"]]
                ft_caps = sta_ft.forward_with_capture(self.ft, [orig]+paras, self.cfg.batch_size, last_only=True)
                p = softmax_logits(ft_caps["LOGITS"][0]).detach()

                for i, para in enumerate(paras):
                    enc = tokenize_batch(tok, [para], self.cfg.device)
                    last_ix = enc["last_index"][0].item()
                    with torch.no_grad():
                        out = self.ft.model(
                            input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"]
                        )
                        logits = out.logits[0, last_ix, :]
                    skl_ft = sym_kl(p, softmax_logits(logits)).item()
                    rows.append({
                        "family_index": fam["index"],
                        "layer": layer_idx,
                        "cond": "FT_baseline",
                        "sKL": skl_ft
                    })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(outdir, "H5_rows.csv"), index=False)
        self.logger.info(f"[H5] Saved {os.path.join(outdir,'H5_rows.csv')} (baseline sKL only; RES swap stub documented in code).")

    # H6
    def run_H6(self):
        self.logger.info("=== H6: Layer specificity (covered by multi-layer runs in H1/H2) ===")
        self.logger.info("No separate pass needed — supply multiple --layers to H1/H2.")

    # Markdown Summary
    def write_markdown_summary(self):
        def pm(m, c):
            if np.isnan(m): return "nan"
            if np.isnan(c): return f"{m:.4f}"
            return f"{m:.4f} ± {c:.4f}"

        lines = []
        lines.append("# Paraphrase Robustness – Statistical Summary\n")
        lines.append(f"- **Base model:** `{self.cfg.base_model}`")
        if self.cfg.ft_model:
            lines.append(f"- **FT model:** `{self.cfg.ft_model}`")
        elif self.cfg.lora_adapter:
            lines.append(f"- **FT:** base+LoRA `{self.cfg.lora_adapter}` (merge={self.cfg.merge_lora})")
        lines.append(f"- **Data:** `{self.cfg.data_json}`")
        lines.append(f"- **Layers:** {self.layers}")
        lines.append(f"- **Heads tested:** {self.heads}")
        lines.append(f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # H1
        h1_dir = os.path.join(self.out, "H1_orientation")
        if os.path.isdir(h1_dir):
            rows = []
            for fn in sorted(os.listdir(h1_dir)):
                if fn.endswith("_rows.csv"):
                    df = pd.read_csv(os.path.join(h1_dir, fn))
                    if df.empty: continue
                    for L, g in df.groupby("layer"):
                        n = int(len(g))
                        rows.append([
                            L, n,
                            pm(g["base_f_sens_mean"].mean(), ci95(g["base_f_sens_mean"].values)),
                            pm(g["ft_f_sens_mean"].mean(),   ci95(g["ft_f_sens_mean"].values)),
                            pm(g["delta_f_sens"].mean(),     ci95(g["delta_f_sens"].values)),
                        ])
            if rows:
                lines.append("## H1 — Orientation change at DOWN\n")
                lines.append("| layer | N families | BASE f_sens | FT f_sens | Δ f_sens (FT-BASE) |")
                lines.append("|:-----:|-----------:|------------:|----------:---:|")
                for r in sorted(rows, key=lambda x: x[0]):
                    lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")
                lines.append("")

        # H2
        h2_dir = os.path.join(self.out, "H2_projection")
        if os.path.isdir(h2_dir):
            rows = []
            for fn in sorted(os.listdir(h2_dir)):
                if fn.endswith("_rows.csv"):
                    df = pd.read_csv(os.path.join(h2_dir, fn))
                    if df.empty: continue
                    for L, g in df.groupby("layer"):
                        n = int(len(g))
                        rows.append([
                            L, n,
                            pm(g["sKL_FT"].mean(),              ci95(g["sKL_FT"].values)),
                            pm(g["sKL_sens_only"].mean(),       ci95(g["sKL_sens_only"].values)),
                            pm(g["sKL_orth_only"].mean(),       ci95(g["sKL_orth_only"].values)),
                            pm(g["delta_sens_minus_FT"].mean(), ci95(g["delta_sens_minus_FT"].values)),
                            pm(g["delta_orth_minus_FT"].mean(), ci95(g["delta_orth_minus_FT"].values)),
                        ])
            if rows:
                lines.append("## H2 — Projection interventions (sens-only vs orth-only)\n")
                lines.append("| layer | N pairs | sKL(FT) | sKL(sens-only) | sKL(orth-only) | Δ(sens−FT) | Δ(orth−FT) |")
                lines.append("|:-----:|--------:|--------:|---------------:|---------------:|-----------:|-----------:|")
                for r in sorted(rows, key=lambda x: x[0]):
                    lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |")
                lines.append("")

        # H3
        h3_dir = os.path.join(self.out, "H3_heads")
        if os.path.isdir(h3_dir):
            abl_path = os.path.join(h3_dir, "H3_ablation_rows.csv")
            if os.path.exists(abl_path):
                df = pd.read_csv(abl_path)
                if not df.empty:
                    lines.append("## H3 — Head ablation (necessity)\n")
                    lines.append("| layer | head | N | mean sKL |")
                    lines.append("|:-----:|:----:|--:|---------:|")
                    for (L, H), g in df.groupby(["layer","head"]):
                        lines.append(f"| {L} | {H} | {len(g)} | {pm(g['sKL'].mean(), ci95(g['sKL'].values))} |")
                    lines.append("")
            trans_path = os.path.join(h3_dir, "H3_transplant_rows.csv")
            if os.path.exists(trans_path):
                df = pd.read_csv(trans_path)
                if not df.empty:
                    lines.append("## H3 — Head transplant (sufficiency)\n")
                    lines.append("| layer | N | mean sKL |")
                    lines.append("|:-----:|--:|---------:|")
                    for L, g in df.groupby("layer"):
                        lines.append(f"| {L} | {len(g)} | {pm(g['sKL'].mean(), ci95(g['sKL'].values))} |")
                    lines.append("")

        # H4
        h4_dir = os.path.join(self.out, "H4_interp")
        h4_path = os.path.join(h4_dir, "H4_interp_rows.csv")
        if os.path.exists(h4_path):
            df = pd.read_csv(h4_path)
            if not df.empty:
                lines.append("## H4 — DOWN interpolation dose–response\n")
                lines.append("| layer | λ | mean sKL ± CI |")
                lines.append("|:-----:|:-:|---------------:|")
                for (L), g in df.groupby("layer"):
                    g = g.sort_values("lambda")
                    for _, r in g.iterrows():
                        lines.append(f"| {int(L)} | {r['lambda']:.2f} | {pm(r['mean_sKL'], r['ci_sKL'])} |")
                lines.append("")

        # H5
        h5_dir = os.path.join(self.out, "H5_style")
        h5_path = os.path.join(h5_dir, "H5_rows.csv")
        if os.path.exists(h5_path):
            df = pd.read_csv(h5_path)
            if not df.empty:
                lines.append("## H5 — Baseline sKL (per layer)\n")
                lines.append("| layer | N | mean sKL |")
                lines.append("|:-----:|--:|---------:|")
                for L, g in df.groupby("layer"):
                    lines.append(f"| {int(L)} | {len(g)} | {pm(g['sKL'].mean(), ci95(g['sKL'].values))} |")
                lines.append("")

        # Write file
        report_path = os.path.join(self.out, "REPORT.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        self.logger.info(f"[REPORT] Wrote summary to {report_path}")

    # Orchestrate
    def run(self):
        self.load_models()
        self.load_data()
        if not self.families:
            self.logger.warning("No prompt families were built.")
        self.logger.info(f"Config: {asdict(self.cfg)}")

        if self.cfg.run_H1: self.run_H1()
        if self.cfg.run_H2: self.run_H2()
        if self.cfg.run_H3: self.run_H3()
        if self.cfg.run_H4: self.run_H4()
        if self.cfg.run_H5: self.run_H5()
        if self.cfg.run_H6: self.run_H6()

        # Always try to write the summary at the end
        try:
            self.write_markdown_summary()
        except Exception as e:
            self.logger.error(f"[REPORT] Failed to write REPORT.md: {e}")

# main

def main():
    cfg = parse_args()
    ensure_dir(cfg.output_dir)
    logger = setup_logging(cfg.output_dir)
    set_seed(cfg.seed)
    runner = Runner(cfg, logger)
    start = time.time()
    runner.run()
    logger.info(f"Done in {time.time()-start:.1f}s. Outputs in: {cfg.output_dir}")

if __name__ == "__main__":
    main()
