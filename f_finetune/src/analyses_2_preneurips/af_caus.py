#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IIT + Causal Scrubbing 

Inputs:
--data_path:  prompts JSON or JSONL (the format; one list or JSONL lines)
--base_model_path: HF path to BASE model
--ft_model_path:   HF path to FT; can be (a) full model dir  OR (b) LoRA adapter dir
--ft_is_lora:      set if --ft_model_path is a LoRA adapter (merged into base for eval)
--score_model_path: (optional) a model used as TF proxy scorer
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as _dt
import gc
import io
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm

def _safe_json_load(path: str) -> List[dict]:
    """
    Read either a JSON array file OR JSONL (one JSON object per line)
    Ignores blank lines. Raises on the first malformed line
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        head = f.read(2)
        f.seek(0)
        if head.lstrip().startswith("["):
            return json.load(f)
        # JSONL
        rows = []
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Bad JSON on line {i} of {path}: {e}")
        return rows


def build_chat_prompt(tokenizer, instruction: str, inp: str = "") -> str:
    user_msg = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def three_way_split(ds: Dataset, *, val_pct: float, test_pct: float, seed: int):
    rng = np.random.default_rng(seed)
    pcs = list({int(ex["prompt_count"]) for ex in ds})
    rng.shuffle(pcs)
    n = len(pcs)
    n_val = int(n * val_pct)
    n_test = int(n * test_pct)
    val_ids  = set(pcs[:n_val])
    test_ids = set(pcs[n_val:n_val+n_test])
    #train_ids = set(pcs[n_val+n_val+n_test:])
    train_ids = set(pcs[n_val+n_test:])
    return train_ids, val_ids, test_ids


def _select_layers_like_gemma(model) -> List[int]:
    """
    Utility: return candidate layer indices {6,10,14,18,22} clipped to model depth
    """
    def _get_attr_chain(obj: Any, chain: str) -> Any:
        cur = obj
        for name in chain.split('.'):
            cur = getattr(cur, name)
        return cur
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    candidates = [
        "model.layers",
        "model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "backbone.layers",
        "layers",
    ]
    layers = None
    for path in candidates:
        try:
            layers = _get_attr_chain(base, path)
            if isinstance(layers, (list, nn.ModuleList)):
                break
        except Exception:
            pass
    if layers is None:
        raise AttributeError("Cannot locate layer list on the model.")
    depth = len(layers)
    wanted = [6, 10, 14, 18, 22]
    return [i for i in wanted if 0 <= i < depth]


def load_model_and_tokenizer_generic(
    base_model_path: str,
    lora_path: Optional[str] = None,
    merge_lora: bool = True,
    quant: str = "none",
    device_map: str = "auto",
):
    import importlib
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs = {}
    try:
        _BNB_OK = importlib.util.find_spec("bitsandbytes") is not None
    except Exception:
        _BNB_OK = False
    if quant != "none" and not _BNB_OK:
        logging.warning("bitsandbytes not available - falling back to bf16")
        quant = "none"

    if quant == "none":
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = device_map
    else:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=(quant == "8bit"),
            load_in_4bit=(quant == "4bit"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = device_map

    logging.info("Loading BASE from %s", base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)

    tok_path = base_model_path
    model = base_model
    if lora_path:
        try:
            from peft import PeftModel
        except Exception:
            logging.error("peft is not installed but --ft_is_lora was given.")
            raise
        logging.info("Loading LoRA adapter from %s", lora_path)
        model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=False)
        if merge_lora:
            logging.info("Merging LoRA weights into base …")
            model = model.merge_and_unload()
            logging.info("LoRA merge complete.")

    tokenizer = AutoTokenizer.from_pretrained(tok_path, model_max_length=4096, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.eval()
    try:
        model = torch.compile(model)  # optional
    except Exception:
        pass
    return model, tokenizer

@dataclasses.dataclass
class NodeSpec:
    layer: int
    module: str   # "attn_out" | "mlp_post" | "block_out"
    tslice: str   # "last" | "mean"
    def tag(self) -> str:
        return f"L{self.layer}:{self.module}:{self.tslice}"

class HookManager:
    """
    Registers hooks on the chosen NodeSpecs and lets us run in different modes:
      - CAPTURE_ORIG: record ORIGINAL activations for each node & sample
      - PATCH: in paraphrase pass, mix paraphrase activation with stored orig via gates g in [0,1]
      - SCRUB: replace *non-masked* nodes with batch-shuffle noise; keep masked nodes patched from ORIG
      - ABLATE: replace *masked* nodes with zeros or matched noise (necessity test)
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        nodes: List[NodeSpec],
        gates: Dict[str, torch.Tensor],  # tag -> scalar gate (parameter tensor)
        device: torch.device,
        noise_stats: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        ablate_zero: bool = False,
        logit_B: Optional[torch.Tensor] = None,
        enforce_logit_nullspace: bool = False,
        scrub_mode: str = "shuffle",  # "shuffle" or "gaussian"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.nodes = nodes
        self.gates = gates
        self.device = device
        self.noise_stats = noise_stats or {}
        self.ablate_zero = ablate_zero

        self.logit_B = logit_B
        self.enforce_logit_nullspace = enforce_logit_nullspace
        self.scrub_mode = scrub_mode

        # Internal buffers per forward
        self.mode = "CAPTURE_ORIG"
        self.batch_ctx = {}  # set by caller: contains attn_mask (+ optional meta)
        self.orig_buf: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []

        self._handles_by_tag: Dict[str, nn.Module] = {}
        self._install_handles()

    def _project_out_logits(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.enforce_logit_nullspace) or (self.logit_B is None):
            return x
        B = self.logit_B.to(dtype=x.dtype, device=x.device)  # [H, r]
        if x.dim() == 2:  # [B, H]
            return x - (x @ B) @ B.T
        elif x.dim() == 3:  # [B, S, H]
            flat = x.reshape(-1, x.size(-1))
            flat = flat - (flat @ B) @ B.T
            return flat.view_as(x)
        else:
            return x

    def _get_layers(self):
        def _get_attr_chain(obj: Any, chain: str) -> Any:
            cur = obj
            for name in chain.split('.'):
                cur = getattr(cur, name)
            return cur
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        for path in ["model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers", "backbone.layers", "layers"]:
            try:
                layers = _get_attr_chain(base, path)
                if isinstance(layers, (list, nn.ModuleList)):
                    return layers
            except Exception:
                pass
        raise AttributeError("Cannot locate transformer layers.")

    def _install_handles(self):
        layers = self._get_layers()
        for ns in self.nodes:
            layer_mod = layers[ns.layer]
            if ns.module == "attn_out":
                mod = getattr(layer_mod, "self_attn", None) or getattr(layer_mod, "attn", None)
                mod = getattr(mod, "o_proj", None) or getattr(layer_mod, "o_proj", None)
            elif ns.module == "mlp_post":
                mlp = getattr(layer_mod, "mlp", None) or getattr(layer_mod, "feed_forward", None)
                mod = getattr(mlp, "down_proj", None)
            elif ns.module == "block_out":
                mod = layer_mod
            else:
                raise ValueError(f"Unknown module type {ns.module}")
            if mod is None:
                logging.warning("Node %s: target submodule not found; will skip.", ns.tag())
                continue
            self._handles_by_tag[ns.tag()] = mod

    @staticmethod
    def _last_token_index(attn_mask: torch.Tensor) -> torch.Tensor:
        return attn_mask.sum(dim=1) - 1  # [B]

    @staticmethod
    def _mean_shift_delta(y: torch.Tensor, mask: torch.Tensor, target_mean: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=y.dtype, device=y.device)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1)
        m = (y * mask_f.unsqueeze(-1)).sum(dim=1) / denom
        delta = (target_mean - m).unsqueeze(1)
        return delta

    # hook kernels
    def _apply_tslice_mix(self, out: torch.Tensor, ns: NodeSpec, tag: str) -> torch.Tensor:
        g = torch.clamp(self.gates[tag].to(dtype=out.dtype, device=out.device), 0.0, 1.0)
        one = torch.ones((), dtype=out.dtype, device=out.device)

        if ns.tslice == "last":
            attn_mask = self.batch_ctx["attn_mask"]
            last_idx = self._last_token_index(attn_mask)
            B, S, H = out.shape
            idx = last_idx.view(B, 1, 1).expand(-1, 1, H)
            cur_last = out.gather(dim=1, index=idx).squeeze(1)  # [B,H]
            target_last = self.orig_buf[tag].to(dtype=out.dtype, device=out.device)
            mixed_last = (one - g) * cur_last + g * target_last
            # project only the patch contribution
            patch_only = mixed_last - cur_last
            patch_only = self._project_out_logits(patch_only)
            mixed_last = cur_last + patch_only
            return out.scatter(dim=1, index=idx, src=mixed_last.unsqueeze(1))
        else:  # 'mean'
            attn_mask = self.batch_ctx["attn_mask"]
            target_mean = self.orig_buf[tag].to(dtype=out.dtype, device=out.device)
            delta = self._mean_shift_delta(out, attn_mask, target_mean)  # [B,1,H]
            delta = self._project_out_logits(delta)
            return out + g * delta

    def _apply_scrub(self, out: torch.Tensor, ns: NodeSpec, tag: str, is_masked_node: bool) -> torch.Tensor:
        """
        SCRUB mode: keep masked nodes patched; scrub others by within-batch token-wise shuffling
        """
        if is_masked_node:
            return self._apply_tslice_mix(out, ns, tag)

        if self.scrub_mode == "gaussian":
            mu, sigma = self.noise_stats.get(tag, (None, None))
            if mu is None or sigma is None:
                return torch.randn_like(out)
            mu = mu.to(dtype=out.dtype, device=out.device)
            sigma = sigma.to(dtype=out.dtype, device=out.device)
            return torch.randn_like(out) * sigma + mu

        # batch-shuffle scrubbing
        y = out
        if y.dim() == 3:
            B, S, H = y.shape
            y_scrub = y.clone()
            for s in range(S):
                perm = torch.randperm(B, device=y.device)
                y_scrub[:, s, :] = y[perm, s, :]
            return y_scrub
        elif y.dim() == 2:
            B, H = y.shape
            perm = torch.randperm(B, device=y.device)
            return y[perm, :]
        else:
            return torch.randn_like(y)

    def _apply_ablate(self, out: torch.Tensor, ns: NodeSpec, tag: str, is_masked_node: bool) -> torch.Tensor:
        """
        ABLATE mode: If node is masked, replace with zeros (or noise fallback); else leave untouched
        """
        if not is_masked_node:
            return out
        if self.ablate_zero:
            if ns.tslice == "last":
                attn_mask = self.batch_ctx["attn_mask"]
                last_idx = self._last_token_index(attn_mask)
                B, S, H = out.shape
                idx = last_idx.view(B, 1, 1).expand(-1, 1, H)
                z = torch.zeros(B, H, dtype=out.dtype, device=out.device)
                return out.scatter(dim=1, index=idx, src=z.unsqueeze(1))
            else:
                return out * 0.0
        # noise ablation (fallback)
        mu, sigma = self.noise_stats.get(tag, (0.0, 1.0))
        if not torch.is_tensor(mu):
            mu = torch.tensor(mu, dtype=out.dtype, device=out.device)
        else:
            mu = mu.to(dtype=out.dtype, device=out.device)
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, dtype=out.dtype, device=out.device)
        else:
            sigma = sigma.to(dtype=out.dtype, device=out.device)
        eps = torch.randn_like(out) * sigma + mu
        return eps

    # registration modes
    def set_mode(self, mode: str, active_mask: Optional[set] = None):
        """
        mode in {"CAPTURE_ORIG","PATCH","SCRUB","ABLATE"}
        active_mask: set of tags considered 'masked/selected' for SCRUB/ABLATE decisions
        """
        self.mode = mode
        self.active_mask = active_mask or set()
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()

        if not self._handles_by_tag:
            return

        for ns in self.nodes:
            tag = ns.tag()
            mod = self._handles_by_tag.get(tag, None)
            if mod is None:
                continue

            def make_hook(ns_local: NodeSpec, tag_local: str):
                def _hook(module, inputs, outputs):
                    out = outputs
                    out0 = out[0] if isinstance(out, tuple) else out
                    y = out0
                    if self.mode == "CAPTURE_ORIG":
                        if ns_local.tslice == "last":
                            attn_mask = self.batch_ctx["attn_mask"]
                            last_idx = self._last_token_index(attn_mask)
                            B, S, H = y.shape
                            idx = last_idx.view(B, 1, 1).expand(-1, 1, H)
                            last_vec = y.gather(1, idx).squeeze(1)  # [B,H]
                            self.orig_buf[tag_local] = last_vec.detach()
                        else:  # "mean"
                            attn_mask = self.batch_ctx["attn_mask"].to(dtype=y.dtype, device=y.device)
                            denom = attn_mask.sum(dim=1, keepdim=True).clamp_min(1)
                            mean_vec = (y * attn_mask.unsqueeze(-1)).sum(dim=1) / denom
                            self.orig_buf[tag_local] = mean_vec.detach()
                        return outputs
                    elif self.mode == "PATCH":
                        mixed = self._apply_tslice_mix(y, ns_local, tag_local)
                        return (mixed,) + out[1:] if isinstance(out, tuple) else mixed
                    elif self.mode == "SCRUB":
                        mixed = self._apply_scrub(y, ns_local, tag_local, tag_local in self.active_mask)
                        return (mixed,) + out[1:] if isinstance(out, tuple) else mixed
                    elif self.mode == "ABLATE":
                        mixed = self._apply_ablate(y, ns_local, tag_local, tag_local in self.active_mask)
                        return (mixed,) + out[1:] if isinstance(out, tuple) else mixed
                    return outputs
                return _hook

#            self.hooks.append(mod.register_forward_hook(make_hook(ns, tag), with_kwargs=False))
            try:
                h = mod.register_forward_hook(make_hook(ns, tag), with_kwargs=False)  # PyTorch ≥2.0
            except TypeError:
                h = mod.register_forward_hook(make_hook(ns, tag))  # Older PyTorch
            self.hooks.append(h)

    def set_batch_ctx(self, attn_mask: torch.Tensor):
        self.batch_ctx = {"attn_mask": attn_mask}

    def close(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()

# metrics

def symmetric_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """
    sKL between two logit distributions (per example)
    Inputs: [B, V] logits
    """
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    p_prob = p.exp()
    q_prob = q.exp()
    kl_pq = (p_prob * (p - q)).sum(dim=-1)
    kl_qp = (q_prob * (q - p)).sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp)


def topk_mass_preservation_loss(p_logits: torch.Tensor, q_logits: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    Encourage q to keep probability mass on p's top-k tokens
    """
    with torch.no_grad():
        topk_idx = torch.topk(p_logits, k=k, dim=-1).indices  # [B, k]
    B = p_logits.size(0)
    arange = torch.arange(B, device=p_logits.device).unsqueeze(-1).expand(-1, k)
    q_sel = q_logits[arange, topk_idx]  # [B,k]
    p_sel = p_logits[arange, topk_idx]
    q_lse = torch.logsumexp(q_sel, dim=-1)
    p_lse = torch.logsumexp(p_sel, dim=-1)
    loss = -(q_lse - p_lse)
    return loss.mean()

# data handling

@dataclasses.dataclass
class Pair:
    prompt_count: int
    input_text: str
    original: str
    paraphrase: str
    paraphrase_key: str


def load_pairs_from_json(
    path: str,
    instruct_types: Optional[List[str]] = None,
) -> List[Pair]:
    """
    Supports two prompt formats per record (JSON or JSONL)
    """
    data = _safe_json_load(path)
    pairs: List[Pair] = []

    n_items = 0
    n_with_top_keys = 0
    n_with_list = 0
    n_paras = 0

    for item in data:
        n_items += 1
        try:
            pc = int(item.get("prompt_count", item.get("id", item.get("pc", -1))))
        except Exception:
            pc = -1
        if pc < 0:
            continue

        inp = item.get("input", "") or item.get("inp", "") or ""
        orig = item.get("instruction_original") or item.get("instruction") or item.get("original_instruction") or ""
        if not orig:
            continue

        # top-level instruct_* keys
        top_keys = [k for k in item.keyif isinstance(k, str) and k.startswith("instruct_")]
        if top_keys:
            n_with_top_keys += 1
            for k in top_keys:
                if instruct_types and k not in instruct_types:
                    continue
                ptxt = item.get(k)
                if isinstance(ptxt, str) and ptxt.strip():
                    pairs.append(Pair(pc, inp, orig, ptxt, k))
                    n_paras += 1

        # "paraphrases" list
        ph = item.get("paraphrases")
        if isinstance(ph, list) and ph:
            n_with_list += 1
            for obj in ph:
                if isinstance(obj, str):
                    k = "instruct_freeform"
                    txt = obj
                elif isinstance(obj, dict):
                    k = obj.get("key") or obj.get("style") or obj.get("name") or "instruct_freeform"
                    txt = obj.get("text") or obj.get("instruction") or obj.get("prompt") or ""
                else:
                    continue
                if instruct_types and k not in instruct_types:
                    continue
                if not isinstance(txt, str) or not txt.strip():
                    continue
                pairs.append(Pair(pc, inp, orig, txt, k))
                n_paras += 1

    logging.info(
        "Input scan: %d records | %d with top-level instruct_* | %d with paraphrases list | %d paraphrases total",
        n_items, n_with_top_keys, n_with_list, n_paras,
    )

    if not pairs:
        logging.error(
            "No paraphrases found. This usually means the file uses 'paraphrases':[{'key','text'}] "
            "or top-level 'instruct_*' keys, but neither matched. "
            "First item (redacted) for debugging:\n%s",
            json.dumps([data[0]] if data else [], indent=2)[:2000]
        )
    else:
        sample = pairs[:3]
        logging.info("Sample pairs: %s", [(p.prompt_count, p.paraphrase_key, p.paraphrase[:40]+"…") for p in sample])

    return pairs

def split_pairs(
    pairs: List[Pair],
    val_pct: float,
    test_pct: float,
    seed: int,
    groupwise: bool = True,
    holdout_para_frac: float = 0.25,
) -> Dict[str, List[Pair]]:
    """
    Two axes of holdout: prompts (by prompt_count) and paraphrase types
    """
    ds = Dataset.from_list([{"prompt_count": p.prompt_count} for p in pairs])
    train_ids, val_ids, test_ids = three_way_split(ds, val_pct=val_pct, test_pct=test_pct, seed=seed)
    rng = random.Random(seed)

    def _by_ids(ids: set[int]) -> List[Pair]:
        return [p for p in pairs if p.prompt_count in ids]

    train_all = _by_ids(train_ids)
    val_all   = _by_ids(val_ids)
    test_all  = _by_ids(test_ids)

    if not groupwise:
        tmp = pairs[:]
        rng.shuffle(tmp)
        n = len(tmp)
        n_val = int(n * val_pct)
        n_test = int(n * test_pct)
        val_all, test_all, train_all = tmp[:n_val], tmp[n_val:n_val+n_test], tmp[n_val+n_val+n_test:]

    valtest = val_all + test_all
    types_in_valtest = sorted({p.paraphrase_key for p in valtest})
    if not types_in_valtest:
        logging.warning("No paraphrase types found in val/test — returning plain splits.")
        return {"train": train_all, "val": val_all, "test": test_all, "held_para_types": []}

    cnt = Counter([p.paraphrase_key for p in valtest])
    types_by_freq = [t for t, _ in cnt.most_common()]
    n_hold_target = max(1, int(len(types_by_freq) * holdout_para_frac))

    held_para = set(types_by_freq[:n_hold_target])
    def _eval_pairs(hset: set[str]) -> Tuple[List[Pair], List[Pair]]:
        v = [p for p in val_all  if p.paraphrase_key in hset]
        t = [p for p in test_all if p.paraphrase_key in hset]
        return v, t

    val_pairs, test_pairs = _eval_pairs(held_para)
    i = n_hold_target
    while not (val_pairs or test_pairs) and i < len(types_by_freq):
        i += 1
        held_para = set(types_by_freq[:i])
        val_pairs, test_pairs = _eval_pairs(held_para)

    if not (val_pairs or test_pairs):
        logging.warning("Eval set would be empty after type holdout — disabling type holdout.")
        return {"train": train_all, "val": val_all, "test": test_all, "held_para_types": []}

    train_pairs = [p for p in train_all if p.paraphrase_key not in held_para]

    return {
        "train": train_pairs,
        "val":   val_pairs,
        "test":  test_pairs,
        "held_para_types": sorted(list(held_para)),
    }

# batching / tokenization

def batch_tokenize_pairs(tokenizer, batch: List[Pair]) -> Dict[str, Any]:
    orig_prompts  = [build_chat_prompt(tokenizer, p.original, p.input_text) for p in batch]
    para_prompts  = [build_chat_prompt(tokenizer, p.paraphrase, p.input_text) for p in batch]

    tok_orig = tokenizer(orig_prompts, return_tensors="pt", padding=True, truncation=True,
                         max_length=tokenizer.model_max_length)
    tok_para = tokenizer(para_prompts, return_tensors="pt", padding=True, truncation=True,
                         max_length=tokenizer.model_max_length)

    attn_o = tok_orig["attention_mask"]
    attn_p = tok_para["attention_mask"]
    return {
        "orig": tok_orig,
        "para": tok_para,
        "attn_o": attn_o,
        "attn_p": attn_p,
    }

# IIT mask (Concrete gates)

class GateBank(nn.Module):
    """
    A small collection of scalar gates (one per NodeSpec) with Concrete/hard-sigmoid relaxation
    """
    def __init__(self, nodes: List[NodeSpec], init: float = 0.2, temperature: float = 2.0):
        super().__init__()
        self.tags = [ns.tag() for ns in nodes]
        init_logit = math.log(init) - math.log(1 - init + 1e-9)
        self.logits = nn.Parameter(torch.full((len(self.tags),), float(init_logit)))
        self.temperature = temperature

    def set_temperature(self, t: float):
        self.temperature = float(t)

    def forward(self) -> Dict[str, torch.Tensor]:
        s = torch.sigmoid(self.logits / max(self.temperature, 1e-6))
        return {tag: s[i] for i, tag in enumerate(self.tags)}

    def as_numpy(self) -> Dict[str, float]:
        s = torch.sigmoid(self.logits.detach().cpu() / max(self.temperature, 1e-6))
        return {tag: float(s[i].item()) for i, tag in enumerate(self.tags)}

# noise stats for scrubbing

def collect_noise_stats(
    model: nn.Module,
    tokenizer,
    nodes: List[NodeSpec],
    data_iter: Iterable[List[Pair]],
    device: torch.device,
    max_batches: int = 50,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Collect per-node mean/std over features using ORIGINAL prompts
    (Used only if you set scrub_mode='gaussian'; shuffle mode doesn't need it.)
    """
    manager = HookManager(model, tokenizer, nodes, gates={}, device=device,
                          logit_B=None, enforce_logit_nullspace=False, scrub_mode="shuffle")
    stats = {ns.tag(): [] for ns in nodes}
    manager.set_mode("CAPTURE_ORIG")
    with torch.no_grad():
        for b_idx, batch in enumerate(tqdm(data_iter, desc="calibrating noise", total=max_batches)):
            if b_idx >= max_batches:
                break
            tok = batch_tokenize_pairs(tokenizer, batch)
            inp = {k: v.to(device) for k, v in tok["orig"].items()}
            attn = tok["attn_o"].to(device)
            manager.set_batch_ctx(attn)
            _ = model(**inp)
            for ns in nodes:
                tag = ns.tag()
                if tag in manager.orig_buf:
                    vec = manager.orig_buf[tag]  # [B,H]
                    stats[tag].append(vec.detach().cpu())
    manager.close()

    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for tag, chunks in stats.items():
        if not chunks:
            continue
        X = torch.cat(chunks, dim=0)  # [N,H]
        mu = X.mean(dim=0)
        sd = X.std(dim=0) + 1e-5
        out[tag] = (mu.to(device), sd.to(device))
    return out

# IIT training loop

def iit_train(
    ft_model: nn.Module,
    tokenizer,
    nodes: List[NodeSpec],
    train_batches: List[List[Pair]],
    device: torch.device,
    n_steps: int = 800,
    lr: float = 5e-2,
    temp_start: float = 2.0,
    temp_end: float = 0.3,
    lmbd_sparse: float = 1e-3,
    tf_k: int = 20,
    max_batches_per_epoch: int = 200,
    log_every: int = 20,
    logit_B: Optional[torch.Tensor] = None,
    enforce_logit_nullspace: bool = False,
) -> Tuple[GateBank, List[Dict[str, float]]]:
    """
    Learn scalar gates for each node to minimize sKL(paraphrase_patched || original) at next-token
    """
    if not nodes:
        raise ValueError("iit_train called with an empty node list; nothing to optimize.")

    gatebank = GateBank(nodes, init=0.2, temperature=temp_start).to(device)
    opt = torch.optim.Adam([gatebank.logits], lr=lr)

    logs = []
    step = 0
    ft_model.eval()

    for epoch in range(1, 1000):
        random.shuffle(train_batches)
        for b_idx, batch in enumerate(train_batches[:max_batches_per_epoch]):
            step += 1
            t = temp_start + (temp_end - temp_start) * (step / max(1, n_steps))
            gatebank.set_temperature(max(temp_end, t))

            with torch.no_grad():
                tok = batch_tokenize_pairs(tokenizer, batch)
                inp_o = {k: v.to(device) for k, v in tok["orig"].items()}
                attn_o = tok["attn_o"].to(device)
                out_o = ft_model(**inp_o).logits  # [B,S,V]
                last_idx_o = attn_o.sum(dim=1) - 1
                B = out_o.size(0)
                ar = torch.arange(B, device=device)
                logits_o = out_o[ar, last_idx_o]  # [B,V]

            gates = gatebank()
            manager = HookManager(ft_model, tokenizer, nodes, gates=gates, device=device,
                                  logit_B=logit_B, enforce_logit_nullspace=enforce_logit_nullspace, scrub_mode="shuffle")

            manager.set_mode("CAPTURE_ORIG")
            manager.set_batch_ctx(attn_o)
            with torch.no_grad():
                _ = ft_model(**inp_o)

            manager.set_mode("PATCH")
            tok_p = batch_tokenize_pairs(tokenizer, batch)
            inp_p = {k: v.to(device) for k, v in tok_p["para"].items()}
            attn_p = tok_p["attn_p"].to(device)
            manager.set_batch_ctx(attn_p)
            out_p = ft_model(**inp_p).logits
            last_idx_p = attn_p.sum(dim=1) - 1
            B = out_p.size(0)
            ar = torch.arange(B, device=device)
            logits_p = out_p[ar, last_idx_p]  # [B,V]
            manager.close()

            skl = symmetric_kl(logits_o, logits_p).mean()
            tf_loss = topk_mass_preservation_loss(logits_o, logits_p, k=tf_k)

            gate_values = torch.stack([gates[ns.tag()] for ns in nodes])
            l0 = gate_values.sum()
            l1 = torch.abs(gatebank.logits).mean() * 1e-4

            # optional: encourage some L6 mass if desired
            l6_mask = torch.tensor([1.0 if ns.layer == 6 else 0.0 for ns in nodes],
                                   device=gate_values.device, dtype=gate_values.dtype)
            l6_total = (gate_values * l6_mask).sum()
            lack_l6 = torch.relu(0.5 - l6_total)
            l6_penalty = 0.5 * lack_l6

            loss = skl + 0.1 * tf_loss + lmbd_sparse * l0 + l1 + l6_penalty

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if (step % log_every) == 0 or step == 1:
                entry = {
                    "step": step,
                    "temperature": gatebank.temperature,
                    "skl": float(skl.item()),
                    "tf_loss": float(tf_loss.item()),
                    "l0": float(l0.item()),
                    "mean_gate": float(gate_values.mean().item()),
                    "max_gate": float(gate_values.max().item()),
                }
                logs.append(entry)
                logging.info("[IIT] %s", entry)

            if step >= n_steps:
                break
        if step >= n_steps:
            break

    return gatebank, logs

# evaluation helpers

@torch.no_grad()
def measure_skl_next_token(model, tokenizer, batch: List[Pair], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (skl, logits_o, logits_p) per example at next-token (no patching)
    """
    tok = batch_tokenize_pairs(tokenizer, batch)
    inp_o = {k: v.to(device) for k, v in tok["orig"].items()}
    inp_p = {k: v.to(device) for k, v in tok["para"].items()}
    attn_o = tok["attn_o"].to(device)
    attn_p = tok["attn_p"].to(device)

    out_o = model(**inp_o).logits
    out_p = model(**inp_p).logits

    last_o = attn_o.sum(dim=1) - 1
    last_p = attn_p.sum(dim=1) - 1
    B = out_o.size(0)
    ar = torch.arange(B, device=device)
    logits_o = out_o[ar, last_o]
    logits_p = out_p[ar, last_p]
    skl = symmetric_kl(logits_o, logits_p)
    return skl, logits_o, logits_p


@torch.no_grad()
def evaluate_with_mask_modes(
    ft_model, tokenizer, nodes, gatebank, batches, device,
    noise_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    active_tags: List[str] | Tuple[str, int],
    logit_B: Optional[torch.Tensor] = None,
    enforce_logit_nullspace: bool = False,
) -> Dict[str, float]:
    if not batches:
        logging.warning("[eval] No held-out batches to evaluate. Returning NaNs.")
        nan = float("nan")
        return {
            "FT_unpatched_sKL": nan,
            "FT_patched_sKL": nan,
            "FT_scrubbed_sKL": nan,
            "FT_ablated_sKL": nan,
            "FT_all_scrub_sKL": nan,
            "FT_random_scrub_sKL": nan,
            # samples placeholders
            "samples": {
                "unpatched": [], "patched": [], "scrubbed": [], "ablated": [], "all_scrub": [], "random_scrub": []
            },
        }

    mask_set = set(active_tags) if not (isinstance(active_tags, tuple) and active_tags and active_tags[0] == "__RANDOM__") else set()
    skl_unpatched, skl_patched, skl_scrubbed, skl_ablated = [], [], [], []
    skl_all_scrub_samples, skl_random_scrub_samples = [], []

    for batch in tqdm(batches, desc="eval held-out"):
        tok = batch_tokenize_pairs(tokenizer, batch)
        inp_o = {k: v.to(device) for k, v in tok["orig"].items()}
        inp_p = {k: v.to(device) for k, v in tok["para"].items()}
        attn_o = tok["attn_o"].to(device)
        attn_p = tok["attn_p"].to(device)

        out_o = ft_model(**inp_o).logits
        out_p = ft_model(**inp_p).logits
        last_idx_o = attn_o.sum(dim=1) - 1
        last_idx_p = attn_p.sum(dim=1) - 1
        B = out_o.size(0)
        ar = torch.arange(B, device=device)
        logits_o = out_o[ar, last_idx_o]
        logits_p = out_p[ar, last_idx_p]
        skl_unpatched += symmetric_kl(logits_o, logits_p).tolist()

        gates = gatebank()

        # CAPTURE ORIGINAL activations (teacher side)
        manager_cap = HookManager(
            ft_model, tokenizer, nodes, gates=gates, device=device,
            logit_B=logit_B, enforce_logit_nullspace=enforce_logit_nullspace, scrub_mode="shuffle"
        )
        manager_cap.set_mode("CAPTURE_ORIG")
        manager_cap.set_batch_ctx(attn_o)
        _ = ft_model(**inp_o)
        manager_cap.close()

        # PATCH: value interchange on PARAPHRASE
        manager_patch = HookManager(
            ft_model, tokenizer, nodes, gates=gates, device=device,
            logit_B=logit_B, enforce_logit_nullspace=enforce_logit_nullspace, scrub_mode="shuffle"
        )
        manager_patch.orig_buf = dict(manager_cap.orig_buf)
        manager_patch.set_mode("PATCH")
        manager_patch.set_batch_ctx(attn_p)
        out_patch = ft_model(**inp_p).logits
        logits_patch = out_patch[ar, last_idx_p]
        skl_patched += symmetric_kl(logits_o, logits_patch).tolist()
        manager_patch.close()

        # SCRUB: keep only masked path, shuffle everything else
        manager_scrub = HookManager(
            ft_model, tokenizer, nodes, gates=gates, device=device,
            logit_B=logit_B, enforce_logit_nullspace=enforce_logit_nullspace, scrub_mode="shuffle"
        )
        manager_scrub.orig_buf = dict(manager_cap.orig_buf)
        manager_scrub.set_mode("SCRUB", active_mask=mask_set)
        manager_scrub.set_batch_ctx(attn_p)
        out_scrub = ft_model(**inp_p).logits
        logits_scrub = out_scrub[ar, last_idx_p]
        skl_scrubbed += symmetric_kl(logits_o, logits_scrub).tolist()
        manager_scrub.close()

        # CONTROL: ALL-SCRUB (keep nothing)
        manager_all = HookManager(
            ft_model, tokenizer, nodes, gates=gates, device=device,
            logit_B=logit_B, enforce_logit_nullspace=enforce_logit_nullspace, scrub_mode="shuffle"
        )
        manager_all.orig_buf = dict(manager_cap.orig_buf)
        manager_all.set_mode("SCRUB", active_mask=set())  # keep nothing
        manager_all.set_batch_ctx(attn_p)
        out_all = ft_model(**inp_p).logits
        logits_all = out_all[ar, last_idx_p]
        skl_all_scrub_samples += symmetric_kl(logits_o, logits_all).tolist()
        manager_all.close()

        # CONTROL: RANDOM size-matched mask
        if isinstance(active_tags, tuple) and len(active_tags) == 2 and active_tags[0] == "__RANDOM__":
            k = int(active_tags[1])
            all_tags = [ns.tag() for ns in nodes]
            rand_set = set(random.sample(all_tags, k=min(k, len(all_tags)))) if k > 0 else set()
            manager_rand = HookManager(
                ft_model, tokenizer, nodes, gates=gates, device=device,
                logit_B=logit_B, enforce_logit_nullspace=enforce_logit_nullspace, scrub_mode="shuffle"
            )
            manager_rand.orig_buf = dict(manager_cap.orig_buf)
            manager_rand.set_mode("SCRUB", active_mask=rand_set)
            manager_rand.set_batch_ctx(attn_p)
            out_rand = ft_model(**inp_p).logits
            logits_rand = out_rand[ar, last_idx_p]
            skl_random_scrub_samples += symmetric_kl(logits_o, logits_rand).tolist()
            manager_rand.close()

        # NECESSITY: ablate the masked path on PARAPHRASE only
        manager_p_abl = HookManager(
            ft_model, tokenizer, nodes, gates=gates, device=device,
            ablate_zero=True, logit_B=logit_B,
            enforce_logit_nullspace=enforce_logit_nullspace, scrub_mode="shuffle"
        )
        manager_p_abl.set_mode("ABLATE", active_mask=mask_set)
        manager_p_abl.set_batch_ctx(attn_p)
        out_p_abl = ft_model(**inp_p).logits
        logits_p_abl = out_p_abl[ar, last_idx_p]
        manager_p_abl.close()

        skl_ablated += symmetric_kl(logits_o, logits_p_abl).tolist()

    to_float = lambda xs: float(np.mean(xs) if xs else float("nan"))
    return {
        "FT_unpatched_sKL": to_float(skl_unpatched),
        "FT_patched_sKL": to_float(skl_patched),
        "FT_scrubbed_sKL": to_float(skl_scrubbed),
        "FT_ablated_sKL": to_float(skl_ablated),
        "FT_all_scrub_sKL": to_float(skl_all_scrub_samples),
        "FT_random_scrub_sKL": to_float(skl_random_scrub_samples),
        "samples": {
            "unpatched": skl_unpatched,
            "patched": skl_patched,
            "scrubbed": skl_scrubbed,
            "ablated": skl_ablated,
            "all_scrub": skl_all_scrub_samples,
            "random_scrub": skl_random_scrub_samples,
        }
    }


@torch.no_grad()
def evaluate_base_with_path(
    base_model, ft_model, tokenizer, nodes, gatebank, batches, device,
    logit_B: Optional[torch.Tensor] = None,
    enforce_logit_nullspace: bool = False,
) -> Dict[str, float]:
    """
    Activation-level portability within BASE:
      - Teacher: BASE ORIGINAL logits
      - Compare: BASE PARAPHRASE (unpatched) vs BASE PARAPHRASE patched with BASE ORIGINAL activations
    """
    if not batches:
        logging.warning("[portability] No held-out batches to evaluate. Returning NaNs.")
        nan = float("nan")
        return {
            "BASE_unpatched_sKL_to_BASEorig": nan,
            "BASE_patched_sKL_to_BASEorig": nan,
            "samples": {"base_unpatched": [], "base_patched": []},
        }

    skl_unpatched = []
    skl_basepatched = []

    for batch in tqdm(batches, desc="portability (BASE+path)"):
        tok = batch_tokenize_pairs(tokenizer, batch)
        inp_o = {k: v.to(device) for k, v in tok["orig"].items()}
        inp_p = {k: v.to(device) for k, v in tok["para"].items()}
        attn_o = tok["attn_o"].to(device)
        attn_p = tok["attn_p"].to(device)

        out_o = base_model(**inp_o).logits
        last_o = attn_o.sum(dim=1) - 1
        B = out_o.size(0)
        ar = torch.arange(B, device=device)
        logits_o_base = out_o[ar, last_o]  # teacher

        out_bu = base_model(**inp_p).logits
        last_p = attn_p.sum(dim=1) - 1
        logits_bu = out_bu[ar, last_p]
        skl_unpatched += symmetric_kl(logits_o_base, logits_bu).tolist()

        gates = gatebank()
        manager_cap = HookManager(
            base_model, tokenizer, nodes, gates=gates, device=device,
            logit_B=logit_B, enforce_logit_nullspace=enforce_logit_nullspace, scrub_mode="shuffle"
        )
        manager_cap.set_mode("CAPTURE_ORIG")
        manager_cap.set_batch_ctx(attn_o)
        _ = base_model(**inp_o)
        manager_cap.close()

        manager_patch = HookManager(
            base_model, tokenizer, nodes, gates=gates, device=device,
            logit_B=logit_B, enforce_logit_nullspace=enforce_logit_nullspace, scrub_mode="shuffle"
        )
        manager_patch.orig_buf = dict(manager_cap.orig_buf)
        manager_patch.set_mode("PATCH")
        manager_patch.set_batch_ctx(attn_p)
        out_bp = base_model(**inp_p).logits
        logits_bp = out_bp[ar, last_p]
        skl_basepatched += symmetric_kl(logits_o_base, logits_bp).tolist()
        manager_patch.close()

    return {
        "BASE_unpatched_sKL_to_BASEorig": float(np.mean(skl_unpatched)),
        "BASE_patched_sKL_to_BASEorig": float(np.mean(skl_basepatched)),
        "samples": {"base_unpatched": skl_unpatched, "base_patched": skl_basepatched},
    }

# utility: batching into chunks

def chunked(lst: List[Any], n: int) -> List[List[Any]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# stats helpers for SUMMARY.md

def _format_duration(seconds: float) -> str:
    try:
        from datetime import timedelta
        return str(timedelta(seconds=int(seconds)))
    except Exception:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

def _bootstrap_ci_mean(xs: List[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float,float]:
    if not xs:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    x = np.array(xs, dtype=np.float64)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(np.mean(x[idx]))
    lo = float(np.percentile(stats, 100*alpha/2))
    hi = float(np.percentile(stats, 100*(1 - alpha/2)))
    return lo, hi

def _bootstrap_ci_mean_diff(x: List[float], y: List[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float,float]:
    if not x or not y:
        return (float("nan"), float("nan"))
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    assert len(x) == len(y)
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = y[idx] - x[idx]
        stats.append(np.mean(d))
    lo = float(np.percentile(stats, 2.5))
    hi = float(np.percentile(stats, 97.5))
    return lo, hi

def _paired_sign_flip_pvalue(x: List[float], y: List[float], n_perm: int = 10000, seed: int = 0) -> float:
    """
    Two-sided paired permutation (sign-flip) test on mean difference.
    H0: mean(y - x) == 0, assuming symmetry of differences.
    Returns p-value.
    """
    if not x or not y:
        return float("nan")
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    assert len(x) == len(y)
    d = y - x
    T_obs = np.mean(d)
    rng = np.random.default_rng(seed)
    # Random sign flips
    cnt = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=d.shape)
        T_perm = np.mean(signs * d)
        if abs(T_perm) >= abs(T_obs):
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)  # add-one smoothing
    return float(p)

def _bootstrap_ci_recover_frac(unpatched: List[float], variant: List[float], n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """
    RecoverFrac = 1 - mean(variant)/mean(unpatched). Returns (point, lo, hi)
    """
    if not unpatched or not variant:
        return (float("nan"), float("nan"), float("nan"))
    u = np.array(unpatched, dtype=np.float64)
    v = np.array(variant, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(u)
    point = 1.0 - float(np.mean(v) / np.mean(u))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        uu = u[idx]; vv = v[idx]
        stats.append(1.0 - float(np.mean(vv) / np.mean(uu)))
    lo = float(np.percentile(stats, 2.5))
    hi = float(np.percentile(stats, 97.5))
    return point, lo, hi

# plots & summary

def make_plots_and_summary(out_dir: Path, history: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Training dynamics plot
    try:
        import matplotlib.pyplot as plt
        logs = history.get("iit_logs", [])
        if logs:
            xs = [r["step"] for r in logs]
            skl = [r["skl"] for r in logs]
            mgate = [r["mean_gate"] for r in logs]
            plt.figure()
            plt.plot(xs, skl, label="train sKL")
            plt.plot(xs, mgate, label="mean gate")
            plt.xlabel("step"); plt.ylabel("value")
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(out_dir / "iit_training.png")
            plt.close()
    except Exception as e:
        logging.warning("Plot failure (iit): %s", e)

    # Budget ladder plot
    try:
        import matplotlib.pyplot as plt
        ladder = history.get("budget_ladder", [])
        if ladder:
            K = [rec["K"] for rec in ladder]
            s1 = [rec["FT_patched_sKL"] for rec in ladder]
            s2 = [rec["FT_scrubbed_sKL"] for rec in ladder]
            s3 = [rec["FT_ablated_sKL"] for rec in ladder]
            plt.figure()
            plt.plot(K, s1, marker="o", label="patched")
            plt.plot(K, s2, marker="o", label="scrubbed")
            plt.plot(K, s3, marker="o", label="ablated")
            plt.gca().invert_xaxis()
            plt.xlabel("K (mask size)"); plt.ylabel("sKL (lower better)")
            plt.grid(True); plt.legend(); plt.tight_layout()
            plt.savefig(out_dir / "budget_ladder.png")
            plt.close()
    except Exception as e:
        logging.warning("Plot failure (ladder): %s", e)

    # Condition bar plot with 95% bootstrap CI
    try:
        import matplotlib.pyplot as plt
        evals = history.get("eval", {})
        samples = evals.get("samples", {})
        labels = ["unpatched", "patched", "scrubbed", "ablated"]
        means = [float(np.mean(samples.get(k, []))) if samples.get(k) else float("nan") for k in labels]
        cis = [_bootstrap_ci_mean(samples.get(k, [])) for k in labels]
        errs_lower = [m - lo if (not math.isnan(m) and not math.isnan(lo)) else float("nan") for m, (lo, hi) in zip(means, cis)]
        errs_upper = [hi - m if (not math.isnan(m) and not math.isnan(hi)) else float("nan") for m, (lo, hi) in zip(means, cis)]
        if any([not math.isnan(m) for m in means]):
            plt.figure()
            x = np.arange(len(labels))
            plt.bar(x, means, yerr=[errs_lower, errs_upper], capsize=5)
            plt.xticks(x, labels)
            plt.ylabel("sKL @ next token (lower is better)")
            plt.title("Held-out conditions (mean ± 95% CI)")
            plt.tight_layout()
            plt.savefig(out_dir / "condition_bars.png")
            plt.close()
    except Exception as e:
        logging.warning("Plot failure (condition bars): %s", e)

    # SUMMARY.md
    summ_md = io.StringIO()
    w = summ_md.write

    # Inputs first
    w("# IIT + Causal Scrubbing: Summary\n\n")
    meta = history.get("meta", {})
    cli_args = history.get("cli_args", {})
    duration_sec = float(meta.get("duration_seconds", float("nan")))
    w("## Inputs\n\n")
    w(f"- **Output directory:** `{meta.get('output_dir','')}`\n")
    # Full CLI args used (values only)
    if cli_args:
        w("\n<details>\n<summary><b>Full CLI arguments (resolved)</b></summary>\n\n")
        try:
            w("```json\n")
            w(json.dumps(cli_args, indent=2))
            w("\n```\n")
        except Exception:
            w("Could not render CLI args.\n")
        w("\n</details>\n")
    w(f"\n- **Total duration:** `{_format_duration(duration_sec)}`\n\n")

    # Existing setup block (kept)
    w("## Setup\n\n")
    # Causal-identifiability run flags
    w(f"- Forbid last-N layers: {meta.get('forbid_last_n', 'n/a')}\n")
    w(f"- Scrub mode: {meta.get('scrub_mode','shuffle')}\n")
    w(f"- Project-out logit space: {meta.get('project_out_logit_space', False)}\n")
    if meta.get('project_out_logit_space', False):
        w(f"  - Logit-basis rank: {meta.get('logit_basis_rank','n/a')}\n")

    w(f"- Data: `{meta.get('data_path','')}`\n")
    w(f"- BASE: `{meta.get('base_model','')}`\n")
    w(f"- FT: `{meta.get('ft_model','')}` (merged LoRA: {meta.get('ft_merged', False)})\n")
    w(f"- Held-out paraphrase types: {', '.join(meta.get('held_para_types', []))}\n")
    w(f"- Candidate nodes: {', '.join(meta.get('candidate_nodes', []))}\n")
    w(f"- Train batches: {meta.get('n_train_batches',0)}, Eval batches: {meta.get('n_eval_batches',0)}, Batch size: {meta.get('batch_size',0)}\n")

    base = history.get("baseline", {})
    w("\n## Baselines (sKL original↔paraphrase @ next token)\n\n")
    w("| Model | sKL (↓) |\n|---|---:|\n")
    w(f"| BASE | {base.get('BASE_sKL', float('nan')):.4f} |\n")
    w(f"| FT (unpatched) | {base.get('FT_sKL', float('nan')):.4f} |\n")

    learned = history.get("learned_mask", {})
    w("\n## Learned Mask (top-K)\n\n")
    w(f"- Final K: **{learned.get('K',0)}**\n")
    w(f"- Tags: {', '.join(learned.get('tags', []))}\n")
    w(f"- Gate values: {json.dumps(learned.get('gates', {}), indent=2)}\n")

    # IIT training snapshot table (sample up to ~20 rows) — kept
    logs_tbl = history.get("iit_logs", [])
    if logs_tbl:
        w("\n## IIT Training Log (snapshot)\n\n")
        w("| step | sKL | mean gate | max gate |\n|---:|---:|---:|---:|\n")
        stride = max(1, len(logs_tbl)//20)
        for rec in logs_tbl[::stride]:
            w(f"| {rec.get('step',0)} | {rec.get('skl',float('nan')):.4f} | {rec.get('mean_gate',float('nan')):.4f} | {rec.get('max_gate',float('nan')):.4f} |\n")

    # Full candidate gate values (for transparency) — kept
    all_g = history.get("all_gates", {})
    if all_g:
        w("\n### All candidate gate values\n\n")
        for tag, val in sorted(all_g.items(), key=lambda kv: kv[1], reverse=True):
            w(f"- {tag}: {val:.4f}\n")

    # Causal tests on held-out — kept table
    evals = history.get("eval", {})
    w("\n## Causal tests on held-out\n\n")
    w("| Condition | sKL (↓) |\n|---|---:|\n")
    w(f"| FT unpatched | {evals.get('FT_unpatched_sKL', float('nan')):.4f} |\n")
    w(f"| FT patched (value-interchange) | {evals.get('FT_patched_sKL', float('nan')):.4f} |\n")
    w(f"| FT scrubbed (only path kept) | {evals.get('FT_scrubbed_sKL', float('nan')):.4f} |\n")
    w(f"| FT ablated (path removed) | {evals.get('FT_ablated_sKL', float('nan')):.4f} |\n")

    # Outcome Overview with significance and RecoverFrac
    try:
        samples = evals.get("samples", {})
        u = samples.get("unpatched", [])
        p = samples.get("patched", [])
        s = samples.get("scrubbed", [])
        a = samples.get("ablated", [])
        N = len(u)
        # Per-condition 95% CI
        def _fmt_mean_ci(vals):
            m = float(np.mean(vals)) if vals else float("nan")
            lo, hi = _bootstrap_ci_mean(vals) if vals else (float("nan"), float("nan"))
            return m, lo, hi
        mu_u, lo_u, hi_u = _fmt_mean_ci(u)
        mu_p, lo_p, hi_p = _fmt_mean_ci(p)
        mu_s, lo_s, hi_s = _fmt_mean_ci(s)
        mu_a, lo_a, hi_a = _fmt_mean_ci(a)

        # Pairwise diffs & tests
        def _row(x, y, label):
            d_lo, d_hi = _bootstrap_ci_mean_diff(x, y)
            d_point = float(np.mean(np.array(y) - np.array(x))) if x and y else float("nan")
            pval = _paired_sign_flip_pvalue(x, y)
            return {"label": label, "delta": d_point, "ci": (d_lo, d_hi), "p": pval}

        rows = [
            _row(u, p, "Patched − Unpatched (should be < 0)"),
            _row(u, s, "Scrubbed − Unpatched (should be < 0)"),
            _row(u, a, "Ablated − Unpatched (should be > 0)"),
        ]

        # RecoverFrac
        rec_p, rec_p_lo, rec_p_hi = _bootstrap_ci_recover_frac(u, p)
        rec_s, rec_s_lo, rec_s_hi = _bootstrap_ci_recover_frac(u, s)

        w("\n## Outcome Overview (with significance)\n\n")
        w(f"- Number of held-out examples: **{N}**\n")
        w("\n### Condition means (95% CI)\n\n")
        w("| Condition | mean sKL | 95% CI |\n|---|---:|---:|\n")
        w(f"| Unpatched | {mu_u:.4f} | [{lo_u:.4f}, {hi_u:.4f}] |\n")
        w(f"| Patched | {mu_p:.4f} | [{lo_p:.4f}, {hi_p:.4f}] |\n")
        w(f"| Scrubbed | {mu_s:.4f} | [{lo_s:.4f}, {hi_s:.4f}] |\n")
        w(f"| Ablated | {mu_a:.4f} | [{lo_a:.4f}, {hi_a:.4f}] |\n")

        w("\n### Pairwise improvements (paired sign-flip test)\n\n")
        w("| Comparison | Δ mean sKL | 95% CI for Δ | p-value |\n|---|---:|---:|---:|\n")
        for r in rows:
            w(f"| {r['label']} | {r['delta']:.4f} | [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}] | {r['p']:.4g} |\n")

        w("\n### Gap closure (RecoverFrac)\n\n")
        w("- RecoverFrac measures fraction of the unpatched gap closed: `1 - mean(variant)/mean(unpatched)`.\n\n")
        w("| Variant | RecoverFrac | 95% CI |\n|---|---:|---:|\n")
        w(f"| Patched | {rec_p:.3f} | [{rec_p_lo:.3f}, {rec_p_hi:.3f}] |\n")
        w(f"| Scrubbed | {rec_s:.3f} | [{rec_s_lo:.3f}, {rec_s_hi:.3f}] |\n")

        # Embed the new plot
        w("\n### Plots\n\n")
        w("![Training dynamics](iit_training.png)\n\n")
        w("![Condition means (95% CI)](condition_bars.png)\n\n")
        w("![Budget ladder](budget_ladder.png)\n\n")

    except Exception as _e:
        w("\n## Outcome Overview\n\n")
        w("Could not compute significance due to missing sample data.\n")

    # Controls — kept
    ctrl = history.get("controls", {})
    if ctrl:
        w("\n## Controls\n\n")
        w("| Control | sKL (↓) |\n|---|---:|\n")
        w(f"| All-scrub (keep nothing) | {ctrl.get('all_scrub',{}).get('FT_all_scrub_sKL', float('nan')):.4f} |\n")
        w(f"| Random mask (size-matched) | {ctrl.get('random_scrub',{}).get('FT_random_scrub_sKL', float('nan')):.4f} |\n")

    # Portability — kept
    port = history.get("portability", {})
    if port:
        w("\n## Portability to BASE\n\n")
        w("| Condition vs BASE-original | sKL (↓) |\n|---|---:|\n")
        w(f"| BASE unpatched | {port.get('BASE_unpatched_sKL_to_BASEorig', float('nan')):.4f} |\n")
        w(f"| BASE + path (activation-level) | {port.get('BASE_patched_sKL_to_BASEorig', float('nan')):.4f} |\n")

    # Remove pre-registered thresholds (original block dropped) and replace with the above outcome overview.

    (out_dir / "SUMMARY.md").write_text(summ_md.getvalue(), encoding="utf-8")

# main CLI

def parse_args():
    p = argparse.ArgumentParser(description="IIT + Causal Scrubbing (minimal circuit) for paraphrase invariance")
    p.add_argument("--data_path", required=True, help="Prompts JSON/JSONL (the format)")
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--ft_model_path", required=True, help="Fine-tuned model dir OR LoRA adapter dir (set --ft_is_lora if adapter)")
    p.add_argument("--ft_is_lora", action="store_true", help="Interpret --ft_model_path as LoRA adapter for the same BASE")
    p.add_argument("--merge_lora", action="store_true", help="Merge LoRA into base (on)")

    p.add_argument("--score_model_path", default=None, help="Optional scoring model (unused by default; TF proxy already uses teacher logits)")

    # Splits / data
    p.add_argument("--val_pct", type=float, default=0.05)
    p.add_argument("--test_pct", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--groupwise_split", action="store_true", help="If set, keep all paraphrases for a prompt in the same split")
    p.add_argument("--holdout_para_frac", type=float, default=0.25, help="Fraction of paraphrase types to hold out entirely from training")
    p.add_argument("--max_train_pairs", type=int, default=5000)
    p.add_argument("--max_eval_pairs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=8)

    # IIT training
    p.add_argument("--iit_steps", type=int, default=800)
    p.add_argument("--iit_lr", type=float, default=5e-2)
    p.add_argument("--temp_start", type=float, default=2.0)
    p.add_argument("--temp_end", type=float, default=0.3)
    p.add_argument("--lmbd_sparse", type=float, default=1e-3)
    p.add_argument("--tf_topk", type=int, default=20)

    # Node set
    p.add_argument("--layers", default="6,10,14,18,22")
    p.add_argument("--modules", default="attn_out,mlp_post,block_out")
    p.add_argument("--slices", default="last,mean")

    # Ladder / pruning
    p.add_argument("--ladder_start_K", type=int, default=20)
    p.add_argument("--ladder_final_K", type=int, default=12)

    # Misc
    p.add_argument("--quant", choices=["none","4bit","8bit"], default="none")
    p.add_argument("--output_dir", required=True)

    # Causal-identifiability toggles
    p.add_argument("--forbid_last_n", type=int, default=2,
                   help="Exclude the last N candidate layers from the mask search (prevents late copy paths).")
    p.add_argument("--project_out_logit_space", action="store_true",
                   help="Project every patched signal onto the LM-head logit nullspace (prevents direct logit steering).")
    p.add_argument("--logit_basis_rank", type=int, default=1024,
                   help="Rank of the LM-head row-space basis for nullspace projection.")

    return p.parse_args()

def _build_logit_basis(model: nn.Module, rank: int = 1024) -> torch.Tensor:
    """
    Returns an H x r column-orthonormal matrix B whose columns span (approximately)
    the row space of the LM head (unembedding). We compute this *robustly* in
    float32 on CPU to avoid missing ops (e.g., geqrf in bf16 on CUDA)
    """
    with torch.no_grad():
        head = getattr(model, "lm_head", None)
        if head is None:
            head = model.get_output_embeddings()
        W = head.weight.detach()            # [V, H]
        H = int(W.shape[1])
        r = int(min(rank, H))

        # Work in float32 on CPU for linear algebra stability/availability
        W_cpu = W.to(dtype=torch.float32, device="cpu")

        # Try fast low-rank PCA/SVD if present; otherwise fall back gracefully
        try:
            # Prefer pca_lowrank when available (older PyTorch has this; it’s fast and stable)
            if hasattr(torch, "pca_lowrank"):
                U, S, V = torch.pca_lowrank(W_cpu, q=r, center=False)  # W ≈ U @ diag(S) @ V^T
                B = V[:, :r].contiguous()                              # [H, r] right singular vectors
            else:
                # Standard full SVD on CPU
                U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)  # W = U @ diag(S) @ Vh
                B = Vh.T[:, :r].contiguous()                             # [H, r]
        except Exception:
            # Final fallback: QR on W^T
            Q, _ = torch.linalg.qr(W_cpu.T, mode="reduced")  # [H, min(H,V)]
            B = Q[:, :r].contiguous()                        # [H, r]

        return B  # caller will .to(device); hooks cast B to x.dtype at use time

def main():
    start_t = time.perf_counter()

    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "run.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("CLI Args:\n%s", json.dumps(vars(args), indent=2))

    # Load data
    all_pairs = load_pairs_from_json(args.data_path)
    splits = split_pairs(
        all_pairs,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        seed=args.seed,
        groupwise=args.groupwise_split,
        holdout_para_frac=args.holdout_para_frac,
    )

    train_pairs = splits["train"][: args.max_train_pairs] if args.max_train_pairs > 0 else splits["train"]
    held_pairs_raw = splits["val"] + splits["test"]
    eval_pairs = held_pairs_raw[: args.max_eval_pairs] if args.max_eval_pairs > 0 else held_pairs_raw

    logging.info("Pairs — train: %d | eval: %d | held paraphrase types: %s",
                 len(train_pairs), len(eval_pairs), splits["held_para_types"])

    # Load models (BASE and FT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, tokenizer = load_model_and_tokenizer_generic(
        base_model_path=args.base_model_path,
        lora_path=None,
        merge_lora=True,
        quant=args.quant,
    )
    if args.ft_is_lora:
        ft_model, _tok2 = load_model_and_tokenizer_generic(
            base_model_path=args.base_model_path,
            lora_path=args.ft_model_path,
            merge_lora=args.merge_lora or True,
            quant=args.quant,
        )
        tokenizer = _tok2
        ft_merged = True
    else:
        ft_model, _tok2 = load_model_and_tokenizer_generic(
            base_model_path=args.ft_model_path,
            lora_path=None,
            merge_lora=True,
            quant=args.quant,
        )
        tokenizer = _tok2
        ft_merged = False

    base_model.to(device)
    ft_model.to(device)

    # Build logit row-space basis once (optional)
    logit_B = None
    if args.project_out_logit_space:
        logit_B = _build_logit_basis(ft_model, rank=args.logit_basis_rank).to(device)

    # Candidate nodes (filter out block_out + forbid last N available layers)
    wanted_layers_user = [int(x) for x in args.layers.split(",") if x.strip()]
    avail_layers = _select_layers_like_gemma(ft_model)
    modules = [m.strip() for m in args.modules.split(",") if m.strip() and m != "block_out"]

    layers_sorted = sorted(avail_layers)
    if args.forbid_last_n > 0 and len(layers_sorted) > args.forbid_last_n:
        allowed_layers = set(layers_sorted[:-args.forbid_last_n])
    else:
        allowed_layers = set(layers_sorted)

    layers = [i for i in wanted_layers_user if i in allowed_layers]
    slices = [s.strip() for s in args.slices.split(",") if s.strip()]
    nodes = [NodeSpec(L, M, S) for L in layers for M in modules for S in slices]

    # Diagnostics
    logging.info("Layers (arg raw): %r", args.layers)
    logging.info("Modules (arg raw): %r", args.modules)
    logging.info("Slices (arg raw): %r", args.slices)
    logging.info("Layers (available): %r", avail_layers)
    logging.info("Layers (selected): %r", layers)
    logging.info("Modules (parsed): %r", modules)
    logging.info("Slices (parsed): %r", slices)
    logging.info("Candidate nodes: %s", ", ".join(n.tag() for n in nodes))

    if not nodes:
        if 6 in avail_layers and {"attn_out","mlp_post"}.issuperset(set(modules) or {"attn_out","mlp_post"}) and ("last" in (slices or ["last"])):
            logging.warning("No nodes parsed; using fallback: [L6:attn_out:last, L6:mlp_post:last]")
            nodes = [NodeSpec(6, "attn_out", "last"), NodeSpec(6, "mlp_post", "last")]
        else:
            raise SystemExit("No candidate nodes constructed. Check --layers/--modules/--slices.")

    cand_tags = {n.tag() for n in nodes}
    logging.info("Candidate nodes (final): %s", ", ".join(sorted(cand_tags)))

    # Baseline sKL
    base_batches = chunked(eval_pairs, args.batch_size)
    with torch.no_grad():
        skl_base_all = []
        skl_ft_all = []
        for batch in tqdm(base_batches[: min(30, len(base_batches))], desc="baselines"):
            skl_b, _, _ = measure_skl_next_token(base_model, tokenizer, batch, device)
            skl_f, _, _ = measure_skl_next_token(ft_model, tokenizer, batch, device)
            skl_base_all += skl_b.tolist()
            skl_ft_all += skl_f.tolist()
    baseline = {
        "BASE_sKL": float(np.mean(skl_base_all)) if skl_base_all else float("nan"),
        "FT_sKL": float(np.mean(skl_ft_all)) if skl_ft_all else float("nan"),
    }
    logging.info("Baseline sKL — BASE: %.4f | FT: %.4f", baseline["BASE_sKL"], baseline["FT_sKL"])

    # Build train/eval batches
    random.shuffle(train_pairs)
    train_batches = chunked(train_pairs, args.batch_size)
    eval_batches = chunked(eval_pairs, args.batch_size)

    # Noise stats (not needed for shuffle scrub, but harmless)
    noise_stats = collect_noise_stats(
        ft_model, tokenizer, nodes, data_iter=train_batches, device=device, max_batches=50
    )

    # IIT train
    gatebank, iit_logs = iit_train(
        ft_model=ft_model,
        tokenizer=tokenizer,
        nodes=nodes,
        train_batches=train_batches,
        device=device,
        n_steps=args.iit_steps,
        lr=args.iit_lr,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        lmbd_sparse=args.lmbd_sparse,
        tf_k=args.tf_topk,
        max_batches_per_epoch=min(len(train_batches), 200),
        log_every=20,
        logit_B=logit_B,
        enforce_logit_nullspace=args.project_out_logit_space,
    )

    # Rank nodes by learned gate strength
    gate_vals = gatebank.as_numpy()  # tag -> value in [0,1]
    ranked = sorted(gate_vals.items(), key=lambda kv: kv[1], reverse=True)

    # Budget ladder
    ladder_start = min(args.ladder_start_K, len(ranked))
    ladder_final = min(args.ladder_final_K, ladder_start)
    Ks = list(range(ladder_start, ladder_final - 1, -2)) + [ladder_final]
    budget_ladder = []
    for K in Ks:
        active = [tag for tag, v in ranked[:K]]
        # Evaluate sufficiency/necessity on held-out
        eval_res = evaluate_with_mask_modes(
            ft_model, tokenizer, nodes, gatebank, eval_batches, device,
            noise_stats=noise_stats, active_tags=active,
            logit_B=logit_B, enforce_logit_nullspace=args.project_out_logit_space
        )

        eval_rec = {"K": K, **eval_res}
        budget_ladder.append(eval_rec)
        logging.info("[LADDER K=%d] %s", K, eval_res)

    # Pick best K by scrubbed sKL (sufficiency under scrubbing)
    best = min(budget_ladder, key=lambda r: r["FT_scrubbed_sKL"])
    bestK = int(best["K"])
    active_final = [tag for tag, v in ranked[:bestK]]

    # Convert gates to a hard mask for eval: set selected tags to 1, others to 0
    hard_gates = {tag: (1.0 if tag in active_final else 0.0) for tag, _ in ranked}
    def gatebank_with_hard_mask_factory(nodes, hard_gates_dict):
        class _HardGateBank(nn.Module):
            def __init__(self, nodes):
                super().__init__()
                self.tags = [ns.tag() for ns in nodes]
            def forward(self):
                return {t: torch.tensor(float(hard_gates_dict.get(t, 0.0))) for t in self.tags}
            def as_numpy(self):
                return {t: float(hard_gates_dict.get(t, 0.0)) for t in self.tags}
        return _HardGateBank(nodes)

    gatebank_hard = gatebank_with_hard_mask_factory(nodes, hard_gates)


    # Final evaluations / portability
    eval_final = evaluate_with_mask_modes(
        ft_model, tokenizer, nodes, gatebank_hard, eval_batches, device,
        noise_stats={},  # not needed with batch-shuffle scrub
        active_tags=active_final
    )

    # Controls: all-scrub (keep nothing) and random size-matched mask
    k_final = len(active_final)
    eval_all_scrub = evaluate_with_mask_modes(
        ft_model, tokenizer, nodes, gatebank_hard, eval_batches, device,
        noise_stats={}, active_tags=[]
    )
    eval_random = evaluate_with_mask_modes(
        ft_model, tokenizer, nodes, gatebank_hard, eval_batches, device,
        noise_stats={}, active_tags=("__RANDOM__", k_final)
    )

    port = evaluate_base_with_path(
        base_model, ft_model, tokenizer, nodes, gatebank_hard, eval_batches, device
    )

    # Save artifacts
    (out_dir / "gate_values.json").write_text(json.dumps(gate_vals, indent=2), encoding="utf-8")
    (out_dir / "budget_ladder.json").write_text(json.dumps(budget_ladder, indent=2), encoding="utf-8")
    learned_mask = {"K": bestK, "tags": active_final, "gates": {k: gate_vals[k] for k in active_final}}
    (out_dir / "learned_mask.json").write_text(json.dumps(learned_mask, indent=2), encoding="utf-8")

    # Summary
    duration_sec = time.perf_counter() - start_t
    history = {
        "meta": {
            "data_path": args.data_path,
            "base_model": args.base_model_path,
            "ft_model": args.ft_model_path,
            "ft_merged": ft_merged,
            "held_para_types": splits["held_para_types"],
            "candidate_nodes": [n.tag() for n in nodes],
            "n_train_batches": len(train_batches),
            "n_eval_batches": len(eval_batches),
            "batch_size": args.batch_size,
            # causal-identifiability run flags we want shown in the summary
            "forbid_last_n": args.forbid_last_n,
            "project_out_logit_space": bool(args.project_out_logit_space),
            "logit_basis_rank": int(args.logit_basis_rank),
            "scrub_mode": "shuffle",
            "duration_seconds": float(duration_sec),
            "output_dir": str(out_dir.resolve()),
        },
        "cli_args": vars(args),
        "baseline": baseline,
        "iit_logs": iit_logs,
        "budget_ladder": budget_ladder,
        "learned_mask": learned_mask,
        "all_gates": gate_vals,           # dump all candidate gates to summary
        "eval": eval_final,
        "portability": port,
        # thresholds removed in favor of significance reporting
    }

    history["controls"] = {
        "all_scrub": eval_all_scrub,
        "random_scrub": eval_random,
    }

    make_plots_and_summary(out_dir, history)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    logging.info("Done. Results in %s", str(out_dir.resolve()))


if __name__ == "__main__":
    main()
