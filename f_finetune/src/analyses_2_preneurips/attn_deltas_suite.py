#!/usr/bin/env python3
"""
SCRIPT B (speed & observability edition)

attn_deltas_suite.py

Additions:
- Batched attention capture ( --batch_size_attn, default 8 ) for large SECOND datasets.
- Rich progress logging: batch counters, elapsed, ETA; per-phase status.
- Extra cheap metrics from the same attention tensors:
  * Attention entropy (per-head), BASE vs FT, plus deltas and plots.
  * Attention span (E|i-j| per-head), BASE vs FT, plus deltas and plots.
  * Instruction-vs-Input mass (requires our split mask), BASE vs FT, plus deltas and plots.
- Overlay/combined plots so related quantities appear in ONE figure with multiple series.

SCRIPT B (speed & observability edition) — FIXED TO CAPTURE ATTENTION

Key fixes:
- Force attention implementation that returns weights (default --attn_impl eager).
- Ensure output_attentions=True on the model **and** per forward call.
- Disable use_cache for the capture forwards (smaller/faster + avoids odd returns).
- **NEW**: Cap sequence length during attention capture with --max_attn_len (default 512) to avoid T^2 blow-ups.
- Keep your batching, logging, extra metrics, and overlay plots intact.
"""
from __future__ import annotations
import argparse, json, logging, math, os, re, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
# Utilities
# ----------------

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def nowts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

def format_eta(done: int, total: int, start_time: float) -> str:
    if done == 0: return "ETA --:--"
    elapsed = time.time() - start_time
    rate = done / max(elapsed, 1e-6)
    remain = (total - done) / max(rate, 1e-6)
    return f"ETA {int(remain//60):02d}:{int(remain%60):02d} (elapsed {int(elapsed//60):02d}:{int(elapsed%60):02d})"

# ----------------
# Data (SECOND prompts)
# ----------------

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
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items: List[Item] = []
    for obj in data:
        pc = int(obj["prompt_count"])
        inp = obj.get("input","") or ""
        paraphrases = {k:v for k,v in obj.items() if k.startswith("instruct_") and isinstance(v, str)}
        items.append(Item(
            prompt_count=pc,
            instruction_original=obj.get("instruction_original", ""),
            paraphrases=paraphrases,
            input_text=inp
        ))
    return items

def build_prompt_text(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return f"{instruction}\n\nInput: {input_text}"
    return instruction

# ----------------
# Tokenisation
# ----------------

@dataclass
class Encoded:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_mask: torch.Tensor
    split_mask_input: Optional[torch.Tensor] = None
    tokens: Optional[List[str]] = None
    true_len: int = 0

def encode_prompt(tokenizer, text: str, device: str, prompt_span: str = "no_bos") -> Encoded:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]
    T = input_ids.shape[0]

    prompt_mask = torch.ones(T, dtype=torch.bool)
    if prompt_span == "no_bos" and T > 0:
        prompt_mask[0] = False
    elif prompt_span == "pre_eos":
        eos_id = tokenizer.eos_token_id
        eos_pos = (input_ids == eos_id).nonzero(as_tuple=True)[0]
        if eos_pos.numel() > 0:
            first_eos = int(eos_pos[0].item())
            prompt_mask[first_eos:] = False

    split_mask_input = None
    if "\n\nInput:" in text:
        try:
            pre, _post = text.split("\n\nInput:", 1)
            enc_pre  = tokenizer(pre, return_tensors="pt", add_special_tokens=True)
            len_pre  = int(enc_pre["input_ids"].shape[1])
            split_mask_input = torch.zeros(T, dtype=torch.bool)
            split_mask_input[len_pre:] = True
        except Exception:
            split_mask_input = None

    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    return Encoded(input_ids=input_ids.to(device),
                   attention_mask=attention_mask.to(device),
                   prompt_mask=prompt_mask.to(device),
                   split_mask_input=split_mask_input.to(device) if split_mask_input is not None else None,
                   tokens=tokens,
                   true_len=T)

# ----------------
# Model access & hooks (batched)
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
        if self.attn is None:
            raise TypeError("Could not access attention submodule.")

class HookHandles:
    def __init__(self): self.handles=[]
    def add(self, h): self.handles.append(h)
    def remove_all(self):
        for h in self.handles:
            try: h.remove()
            except Exception: pass
        self.handles.clear()

def _pad_batch(tensors: List[torch.Tensor], pad_val: int, device: str) -> torch.Tensor:
    max_len = max(int(t.shape[0]) for t in tensors)
    padded = torch.full((len(tensors), max_len), pad_val, dtype=tensors[0].dtype, device=device)
    for i, t in enumerate(tensors):
        L = int(t.shape[0])
        padded[i, :L] = t
    return padded

def capture_attn_batched(model, layer_index: int, *, force_output_attn: bool = True, max_attn_len: int = 512):
    """
    Returns a callable that, given a batch of Encoded samples, runs **one forward** and
    returns a single tensor of shape [B, H, T, T] for that layer.
    """
    block = BlockAccessor(model, layer_index).block
    attn = getattr(block, "self_attn", None) or getattr(block, "attention", None)
    hooks = HookHandles()
    bucket = {"attn": None}

    def attn_forward_hook(module, inputs, outputs):
        # Expect outputs like (attn_out, attn_probs) if output_attentions=True
        probs = None
        if isinstance(outputs, tuple):
            if len(outputs) >= 2 and isinstance(outputs[1], torch.Tensor):
                probs = outputs[1]
        if probs is None and hasattr(module, "attn_probs"):
            p = getattr(module, "attn_probs")
            if isinstance(p, torch.Tensor):
                probs = p
        if probs is not None:
            # >>> CRITICAL FIX: upcast to fp32 on CPU to avoid NaNs with CPU fp16 ops
            bucket["attn"] = probs.detach().to(torch.float32).cpu()

    hooks.add(attn.register_forward_hook(attn_forward_hook))

    @torch.inference_mode()
    def runner(enc_batch: List[Encoded], tokenizer_pad_id: int, device: str) -> Optional[torch.Tensor]:
        bucket["attn"] = None
        if len(enc_batch) == 0:
            return None

        # Truncate each sequence to max_attn_len to avoid T^2 blow-ups
        ids_list, am_list = [], []
        for e in enc_batch:
            T = int(e.input_ids.shape[0])
            cap = min(T, max_attn_len) if max_attn_len and max_attn_len > 0 else T
            ids_list.append(e.input_ids[:cap])
            am_list.append(e.attention_mask[:cap])

        ids = _pad_batch(ids_list, tokenizer_pad_id, device)
        am  = _pad_batch(am_list, 0, device)

        kwargs = {}
        if force_output_attn:
            kwargs["output_attentions"] = True
        kwargs["use_cache"] = False
        _ = model(input_ids=ids, attention_mask=am, **kwargs)
        attn_prob = bucket["attn"]
        hooks.remove_all()
        return attn_prob  # [B,H,T,T] or None
    runner.hooks = hooks
    return runner

# ----------------
# Robust loader (merged FT or LoRA)
# ----------------

def _is_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").exists() or \
           len(list(path.glob("adapter_model*.bin"))) > 0 or \
           len(list(path.glob("adapter_model*.safetensors"))) > 0

def _is_merged_model_dir(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    has_weights = any((path / n).exists() for n in ["pytorch_model.bin", "pytorch_model.pt", "model.bin"]) \
                  or len(list(path.glob("pytorch_model-*.bin"))) > 0 \
                  or (path / "model.safetensors").exists() \
                  or len(list(path.glob("model-*.safetensors"))) > 0
    return has_weights

def _try_fix_common_typo(p: Path) -> Optional[Path]:
    if p.name.startswith("m") and len(p.name) > 1:
        cand = p.with_name(p.name[1:])
        if cand.exists():
            return cand
    return None

def _resolve_local_adapter_path(path_str: str) -> Optional[Path]:
    if path_str is None:
        return None
    p = Path(path_str)
    if p.exists():
        return p
    cand = _try_fix_common_typo(p)
    if cand and cand.exists():
        return cand
    return None

def _force_attn_impl(model, impl: str):
    impl = impl.lower()
    if impl == "auto":
        return
    try:
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation(impl)
        elif hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = impl
    except Exception:
        pass

def build_model_and_tokenizer(
    base_model_name_or_path: str,
    ft_model_name_or_path: Optional[str],
    ft_lora_adapter: Optional[str],
    device: str,
    *,
    attn_impl: str = "eager"
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    base = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device).eval()
    _force_attn_impl(base, attn_impl)
    if hasattr(base.config, "output_attentions"):
        base.config.output_attentions = True

    if ft_model_name_or_path is None and ft_lora_adapter is None:
        raise ValueError("Provide --ft_model_name_or_path (merged FT) OR --ft_lora_adapter (LoRA adapter dir).")

    if ft_lora_adapter is not None:
        local_path = _resolve_local_adapter_path(ft_lora_adapter)
        if local_path is not None:
            if _is_adapter_dir(local_path):
                if not _HAS_PEFT:
                    raise RuntimeError("peft not installed but --ft_lora_adapter was provided.")
                print(f"[loader] Loading BASE + LoRA adapter from local dir: {local_path}")
                ft = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
                ft = PeftModel.from_pretrained(ft, str(local_path))
                ft = ft.merge_and_unload().eval()
                _force_attn_impl(ft, attn_impl)
                if hasattr(ft.config, "output_attentions"):
                    ft.config.output_attentions = True
                return base, ft, tokenizer
            if _is_merged_model_dir(local_path):
                print(f"[loader] Detected merged FT model at --ft_lora_adapter path. Using merged FT: {local_path}")
                ft = AutoModelForCausalLM.from_pretrained(str(local_path), torch_dtype=dtype).to(device).eval()
                _force_attn_impl(ft, attn_impl)
                if hasattr(ft.config, "output_attentions"):
                    ft.config.output_attentions = True
                return base, ft, tokenizer

    if ft_model_name_or_path is None:
        raise ValueError("When --ft_lora_adapter is not a local adapter dir, you must provide --ft_model_name_or_path.")
    ft = AutoModelForCausalLM.from_pretrained(ft_model_name_or_path, torch_dtype=dtype).to(device).eval()
    _force_attn_impl(ft, attn_impl)
    if hasattr(ft.config, "output_attentions"):
        ft.config.output_attentions = True
    return base, ft, tokenizer

# ----------------
# Metrics
# ----------------

def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    return (a * b).sum(dim=-1)

def symmetric_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    kl_pq = (p * (p / q).log()).sum()
    kl_qp = (q * (q / p).log()).sum()
    return float((kl_pq + kl_qp).item())

# ----------------
# Extra cheap metrics
# ----------------

def attn_entropy(p: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    p = p.clamp(min=eps)
    H = -(p * p.log()).sum(dim=-1).mean()
    return H

def attn_span(p: torch.Tensor) -> float:
    T = p.shape[-1]
    idx = torch.arange(T, dtype=p.dtype, device=p.device)
    D = (idx[None, :] - idx[:, None]).abs()
    row_vals = (p * D).sum(dim=-1)
    return float(row_vals.mean().item())

def mass_on_mask(p: torch.Tensor, mask: torch.Tensor) -> float:
    if mask is None or mask.numel() == 0:
        return float('nan')
    cols = mask.to(p.device)
    return float(p[:, cols].sum().item() / max(p.numel() / p.shape[1], 1.0))

# ----------------
# Global attention similarity (BATCHED)
# ----------------

def run_global_attention_similarity(
    items: List[Item],
    base: nn.Module,
    ft: nn.Module,
    tokenizer,
    layer_index: int,
    device: str,
    outdir: Path,
    prompt_span: str,
    keys: Optional[List[str]],
    keys_regex: Optional[str],
    max_items: int,
    batch_size_attn: int,
    metric_kl: bool = True,
    log: Optional[logging.Logger] = None,
    *,
    max_attn_len: int = 512,
) -> pd.DataFrame:
    log = log or logging.getLogger("attn_deltas")

    t_start = time.time()
    flat: List[Tuple[int, str, Encoded]] = []
    for item in items:
        variants = item.all_prompt_variants(include_original=True, keys=keys, regex=keys_regex)
        for key, text in variants:
            if len(flat) >= max_items:
                break
            enc = encode_prompt(tokenizer, build_prompt_text(text, item.input_text), device, prompt_span=prompt_span)
            flat.append((item.prompt_count, key, enc))
        if len(flat) >= max_items:
            break
    N = len(flat)
    if N == 0:
        return pd.DataFrame()
    log.info("[%s] Global phase: prepared %d (prompt,key) items. Batch size=%d", nowts(), N, batch_size_attn)

    runner_base = capture_attn_batched(base, layer_index, force_output_attn=True, max_attn_len=max_attn_len)
    runner_ft   = capture_attn_batched(ft,   layer_index, force_output_attn=True, max_attn_len=max_attn_len)

    rows = []
    last_log = time.time()

    for start in range(0, N, batch_size_attn):
        end = min(start + batch_size_attn, N)
        batch = flat[start:end]
        encs  = [t[2] for t in batch]

        Ab = runner_base(encs, tokenizer.pad_token_id, device)  # [B,H,T,T] or None
        Af = runner_ft(encs,   tokenizer.pad_token_id, device)
        if Ab is None or Af is None:
            log.warning("Batch %d returned no attn. Likely due to unsupported attn implementation. "
                        "Try --attn_impl eager or sdpa.", start // batch_size_attn)
            continue

        B, H, _, Tcap = Ab.shape
        log.info("Processed global batch %d/%d (captured T=%d, H=%d, B=%d)",
                 (start // batch_size_attn) + 1, (N + batch_size_attn - 1)//batch_size_attn, Tcap, H, B)

        for b in range(B):
            pc, key, enc = batch[b]
            T = min(int(enc.true_len), Ab.shape[-1])
            Pb = Ab[b, :, :T, :T]
            Pf = Af[b, :, :T, :T]

            m = enc.prompt_mask[:T].cpu()
            idx = m.nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            Pb = Pb[:, idx][:, :, idx].contiguous()
            Pf = Pf[:, idx][:, :, idx].contiguous()

            cos_list, skl_list = [], []
            ent_b = []; ent_f = []
            span_b = []; span_f = []
            mass_inp_b = []; mass_inp_f = []

            target_mask_input = None
            if enc.split_mask_input is not None:
                target_mask_input = enc.split_mask_input[:T][idx].cpu()

            for h in range(H):
                bmat = Pb[h]; fmat = Pf[h]
                bv = bmat.reshape(-1); fv = fmat.reshape(-1)
                cos_list.append(float(cosine(bv.unsqueeze(0), fv.unsqueeze(0)).item()))
                if metric_kl:
                    pb = bmat / (bmat.sum() + 1e-8)
                    pf = fmat / (fmat.sum() + 1e-8)
                    skl_list.append(symmetric_kl(pb, pf))

                ent_b.append(float(attn_entropy(bmat).item()))
                ent_f.append(float(attn_entropy(fmat).item()))
                span_b.append(attn_span(bmat))
                span_f.append(attn_span(fmat))
                if target_mask_input is not None and target_mask_input.any():
                    mass_inp_b.append(mass_on_mask(bmat, target_mask_input))
                    mass_inp_f.append(mass_on_mask(fmat, target_mask_input))
                else:
                    mass_inp_b.append(float('nan'))
                    mass_inp_f.append(float('nan'))

            row = {"prompt_count": pc, "key": key}
            for h in range(H): row[f"cos_head{h}"] = cos_list[h]
            row["cos_mean"] = float(np.nanmean(cos_list))
            if metric_kl:
                for h in range(H): row[f"skl_head{h}"] = skl_list[h]
                row["skl_mean"] = float(np.nanmean(skl_list))
            for h in range(H):
                row[f"entropy_base_h{h}"] = ent_b[h]; row[f"entropy_ft_h{h}"] = ent_f[h]
                row[f"span_base_h{h}"] = span_b[h];   row[f"span_ft_h{h}"]   = span_f[h]
                row[f"massInput_base_h{h}"] = mass_inp_b[h]; row[f"massInput_ft_h{h}"] = mass_inp_f[h]
            rows.append(row)

        if (time.time() - last_log) > 5.0:
            done = end
            log.info("[%s] Global batch %d/%d (%d/%d items). %s",
                     nowts(), (start // batch_size_attn) + 1, (N + batch_size_attn - 1)//batch_size_attn,
                     done, N, format_eta(done, N, t_start))
            last_log = time.time()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # quick sanity: count NaNs (helps diagnose blank plots)
    nan_frac = df.isna().mean().mean()
    logging.info("Global DF NaN fraction: %.3f", nan_frac)

    csv_path = outdir / f"attn_similarity_layer{layer_index}.csv"
    df.to_csv(csv_path, index=False)
    log.info("Global CSV written: %s  (rows=%d)", csv_path, len(df))

    # ---- Plots ----
    def safe_vals(values: np.ndarray) -> np.ndarray:
        # replace inf/nan to keep plots from being blank
        v = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return v

    head_cols = [c for c in df.columns if re.match(r"^cos_head\d+$", c)]
    H = len(head_cols)
    head_ids = [int(c.split("head")[1]) for c in head_cols]

    def mean_ci(colnames: List[str]) -> Tuple[List[float], List[float]]:
        means, cis = [], []
        for c in colnames:
            vals = safe_vals(df[c].values.astype(float))
            if len(vals) == 0:
                means.append(0.0); cis.append(0.0); continue
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
            ci = 1.96 * s / math.sqrt(len(vals)) if len(vals)>1 else 0.0
            means.append(m); cis.append(ci)
        return means, cis

    # Cosine per head
    if head_cols:
        fig, ax = plt.subplots(figsize=(max(8, H*0.6), 4))
        cos_means, cos_cis = mean_ci(head_cols)
        ax.bar([f"h{h}" for h in head_ids], cos_means, yerr=cos_cis)
        ax.set_ylim(0,1)
        ax.set_title(f"Attention cosine similarity (BASE vs FT) per head — layer {layer_index}")
        ax.set_ylabel("cosine similarity")
        fig.tight_layout()
        fig.savefig(outdir / f"attn_similarity_layer{layer_index}_cosine_per_head.png", dpi=180)
        plt.close(fig)

        # Heatmap
        fig, ax = plt.subplots(figsize=(max(6, H*0.5), max(6, len(df)*0.25)))
        im = ax.imshow(safe_vals(df[head_cols].values.astype(float)), aspect='auto', interpolation='nearest')
        ax.set_title(f"Attention cosine similarity — all items (layer {layer_index})")
        ax.set_xlabel("head"); ax.set_ylabel("item")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(outdir / f"attn_similarity_layer{layer_index}_cosine_heatmap.png", dpi=180)
        plt.close(fig)

    skl_cols = [c for c in df.columns if re.match(r"^skl_head\d+$", c)]
    if skl_cols:
        ids = [int(c.split("head")[1]) for c in skl_cols]
        skl_means, skl_cis = mean_ci(skl_cols)
        fig, ax = plt.subplots(figsize=(max(8, len(ids)*0.6), 4))
        ax.bar([f"h{h}" for h in ids], skl_means, yerr=skl_cis)
        ax.set_title(f"Attention symmetric KL (BASE vs FT) per head — layer {layer_index}")
        ax.set_ylabel("symmetric KL")
        fig.tight_layout()
        fig.savefig(outdir / f"attn_similarity_layer{layer_index}_skl_per_head.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(6, len(ids)*0.5), max(6, len(df)*0.25)))
        im = ax.imshow(safe_vals(df[skl_cols].values.astype(float)), aspect='auto', interpolation='nearest')
        ax.set_title(f"Attention symmetric KL — all items (layer {layer_index})")
        ax.set_xlabel("head"); ax.set_ylabel("item")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(outdir / f"attn_similarity_layer{layer_index}_skl_heatmap.png", dpi=180)
        plt.close(fig)

    # Entropy / span / input-mass overlays and deltas
    ent_b_cols = [f"entropy_base_h{h}" for h in head_ids]
    ent_f_cols = [f"entropy_ft_h{h}"  for h in head_ids]
    span_b_cols = [f"span_base_h{h}" for h in head_ids]
    span_f_cols = [f"span_ft_h{h}"  for h in head_ids]
    mi_b_cols   = [f"massInput_base_h{h}" for h in head_ids]
    mi_f_cols   = [f"massInput_ft_h{h}"   for h in head_ids]

    def means(cols):
        if not cols: return []
        vals = [safe_vals(df[c].values.astype(float)).mean() if c in df else 0.0 for c in cols]
        return [float(v) for v in vals]

    entB, entF = means(ent_b_cols), means(ent_f_cols)
    spanB, spanF = means(span_b_cols), means(span_f_cols)
    miB, miF = means(mi_b_cols), means(mi_f_cols)

    if entB and entF:
        fig, ax = plt.subplots(figsize=(max(8, H*0.6), 4))
        ax.plot([f"h{h}" for h in head_ids], entB, marker="o", label="BASE")
        ax.plot([f"h{h}" for h in head_ids], entF, marker="o", label="FT")
        ax.set_title(f"Attention entropy per head — layer {layer_index}"); ax.set_ylabel("mean entropy"); ax.legend()
        fig.tight_layout(); fig.savefig(outdir / f"attn_entropy_overlay_layer{layer_index}.png", dpi=180); plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(8, H*0.6), 4))
        delta = (np.array(entF) - np.array(entB)).tolist()
        ax.bar([f"h{h}" for h in head_ids], delta)
        ax.set_title(f"Entropy Δ (FT−BASE) per head — layer {layer_index}"); ax.set_ylabel("Δ entropy")
        fig.tight_layout(); fig.savefig(outdir / f"attn_entropy_delta_layer{layer_index}.png", dpi=180); plt.close(fig)

    if spanB and spanF:
        fig, ax = plt.subplots(figsize=(max(8, H*0.6), 4))
        ax.plot([f"h{h}" for h in head_ids], spanB, marker="o", label="BASE")
        ax.plot([f"h{h}" for h in head_ids], spanF, marker="o", label="FT")
        ax.set_title(f"Attention span (E|i−j|) per head — layer {layer_index}"); ax.set_ylabel("mean |i−j|"); ax.legend()
        fig.tight_layout(); fig.savefig(outdir / f"attn_span_overlay_layer{layer_index}.png", dpi=180); plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(8, H*0.6), 4))
        delta = (np.array(spanF) - np.array(spanB)).tolist()
        ax.bar([f"h{h}" for h in head_ids], delta)
        ax.set_title(f"Span Δ (FT−BASE) per head — layer {layer_index}"); ax.set_ylabel("Δ mean |i−j|")
        fig.tight_layout(); fig.savefig(outdir / f"attn_span_delta_layer{layer_index}.png", dpi=180); plt.close(fig)

    if miB and miF:
        fig, ax = plt.subplots(figsize=(max(8, H*0.6), 4))
        ax.plot([f"h{h}" for h in head_ids], miB, marker="o", label="BASE")
        ax.plot([f"h{h}" for h in head_ids], miF, marker="o", label="FT")
        ax.set_title(f"Attention mass to Input: (targets) per head — layer {layer_index}"); ax.set_ylabel("mass"); ax.legend()
        fig.tight_layout(); fig.savefig(outdir / f"attn_mass_input_overlay_layer{layer_index}.png", dpi=180); plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(8, H*0.6), 4))
        delta = (np.array(miF) - np.array(miB)).tolist()
        ax.bar([f"h{h}" for h in head_ids], delta)
        ax.set_title(f"Mass to Input: Δ (FT−BASE) per head — layer {layer_index}"); ax.set_ylabel("Δ mass")
        fig.tight_layout(); fig.savefig(outdir / f"attn_mass_input_delta_layer{layer_index}.png", dpi=180); plt.close(fig)

    return df

# ----------------
# Prompt-level deep dive (unchanged logic; uses same fp32 capture)
# ----------------

def tokens_short(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        s = t.replace("▁", " ").strip()
        if len(s) > 18: s = s[:15] + "…"
        #out.append(s if s else "␠")
        out.append(s if s else "[SP]")
    return out

def run_prompt_deep_dives(
    items: List[Item],
    base: nn.Module,
    ft: nn.Module,
    tokenizer,
    layer_index: int,
    device: str,
    outdir: Path,
    prompt_span: str,
    keys: Optional[List[str]],
    keys_regex: Optional[str],
    prompt_ids: List[int],
    n_paraphrases: int = 8,
    include_original: bool = False,
    select_heads_topk: int = 6,
    max_tokens: int = 96,
):
    for pid in prompt_ids:
        logging.info("[%s] Deep-dive prompt %s", nowts(), pid)
        matches = [it for it in items if it.prompt_count == pid]
        if not matches:
            logging.warning("prompt_count=%s not found in dataset; skipping deep dive.", pid)
            continue
        item = matches[0]
        variants = item.all_prompt_variants(include_original=include_original, keys=keys, regex=keys_regex)
        if len(variants) == 0:
            logging.warning("prompt_count=%s has no matching paraphrases for keys/regex; skipping.", pid)
            continue
        variants = variants[:n_paraphrases]

        encs = [encode_prompt(tokenizer, build_prompt_text(txt, item.input_text), device, prompt_span=prompt_span)
                for _, txt in variants]

        rb = capture_attn_batched(base, layer_index, force_output_attn=True, max_attn_len=512)
        rf = capture_attn_batched(ft,   layer_index, force_output_attn=True, max_attn_len=512)
        Ab = rb(encs, tokenizer.pad_token_id, device)
        Af = rf(encs, tokenizer.pad_token_id, device)
        if Ab is None or Af is None:
            logging.warning("prompt_count=%s: attention not captured; try --attn_impl eager.", pid)
            continue

        B, H, _, _ = Ab.shape
        ddir = ensure_dir(outdir / f"prompt_{pid}")

        head_change = np.zeros(H, dtype=np.float64)
        for b in range(B):
            T = min(int(encs[b].true_len), Ab.shape[-1], max_tokens)
            m = encs[b].prompt_mask[:T]
            idx = m.nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            idx = idx.cpu()
            Pb = Ab[b, :, :T, :T][:, idx][:, :, idx]
            Pf = Af[b, :, :T, :T][:, idx][:, :, idx]
            for h in range(H):
                sim = float(cosine(Pb[h].reshape(-1).unsqueeze(0), Pf[h].reshape(-1).unsqueeze(0)).item())
                head_change[h] += (1.0 - sim)
        if B > 0:
            head_change /= max(B, 1)
        head_rank = np.argsort(-head_change)

        df_rank = pd.DataFrame({"head": np.arange(H, dtype=int), "avg_delta": head_change}).sort_values("avg_delta", ascending=False)
        df_rank.to_csv(ddir / "head_rank_by_change.csv", index=False)

        stats_rows = []
        for b in range(B):
            T = min(int(encs[b].true_len), Ab.shape[-1], max_tokens)
            avgB = Ab[b, :, :T, :T].mean(0)
            avgF = Af[b, :, :T, :T].mean(0)
            row = {"key": variants[b][0]}
            row["BASE_mass_first"] = avgB[:,0].mean().item()
            row["BASE_mass_last"]  = avgB[:,-1].mean().item()
            row["BASE_mass_diag"]  = torch.diag(avgB).mean().item()
            row["FT_mass_first"] = avgF[:,0].mean().item()
            row["FT_mass_last"]  = avgF[:,-1].mean().item()
            row["FT_mass_diag"]  = torch.diag(avgF).mean().item()
            stats_rows.append(row)
        pd.DataFrame(stats_rows).to_csv(ddir / "head_stats.csv", index=False)

        top_heads = head_rank[:select_heads_topk].tolist()
        labels = [k for (k,_) in variants]

        for h in top_heads:
            max_T = max(min(int(encs[i].true_len), Ab.shape[-1], max_tokens) for i in range(B))
            fig, axes = plt.subplots(B, 1, figsize=(max(6, max_T*0.2), max(3, 2*B)), squeeze=False)

            for i in range(B):
                T = min(int(encs[i].true_len), Ab.shape[-1], max_tokens)
                Pb = Ab[i, h, :T, :T]
                Pf = Af[i, h, :T, :T]
                ax = axes[i,0]
                im = ax.imshow((Pf - Pb)[:T,:T], aspect='auto', interpolation='nearest')
                ax.set_title(f"{labels[i]} — head {h} (FT−BASE)")
                tok_i = (encs[i].tokens or [])[:T]
                tt_i  = tokens_short(tok_i)
                ax.set_xticks(range(T)); ax.set_xticklabels(tt_i, rotation=90, fontsize=6)
                ax.set_yticks(range(T)); ax.set_yticklabels(tt_i, fontsize=6)
                fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
            fig.tight_layout()
            fig.savefig(ddir / f"head{h}_attn_grid.png", dpi=170)
            plt.close(fig)

        for h in top_heads:
            rows = []
            for i in range(B):
                T = min(int(encs[i].true_len), Ab.shape[-1], max_tokens)
                # Mean over source positions -> per-target attention vector of length T
                vals = Af[i, h, :T, :T].mean(0)
                k = min(10, int(vals.numel()))
                if k <= 0:
                    topk = []
                else:
                    topk = torch.topk(vals, k=k).indices.tolist()
                tok_strs = [encs[i].tokens[j] for j in topk] if encs[i].tokens else []
                rows.append({"key": labels[i], "top_targets": "|".join(tokens_short(tok_strs))})
            pd.DataFrame(rows).to_csv(ddir / f"head{h}_top_targets_across_paraphrases.csv", index=False)

# ----------------
# Main
# ----------------

def main():
    ap = argparse.ArgumentParser(description="Attention-deltas suite (batched global + prompt-level deep dives)")
    ap.add_argument("--instructions_json", type=str, required=True)
    ap.add_argument("--base_model_name_or_path", type=str, required=True)
    ap.add_argument("--ft_model_name_or_path", type=str, default=None)
    ap.add_argument("--ft_lora_adapter", type=str, default=None)

    ap.add_argument("--layer_index", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--prompt_span", type=str, default="no_bos", choices=["no_bos", "all", "pre_eos"])
    ap.add_argument("--keys", type=str, default=None,
                    help="Comma-separated list of instruct_* keys to include. Default: all instruct_*")
    ap.add_argument("--keys_regex", type=str, default=None, help="Regex to filter instruct_* keys.")
    ap.add_argument("--max_items_global", type=int, default=300, help="Upper bound on (prompt,key) items for global stats.")
    ap.add_argument("--batch_size_attn", type=int, default=8, help="Batch size for attention capture (global and deep-dives).")

    ap.add_argument("--attn_impl", type=str, default="eager", choices=["eager","sdpa","auto"],
                    help="Attention backend to use during capture. 'eager' recommended; FA2 does not return weights.")
    ap.add_argument("--max_attn_len", type=int, default=512,
                    help="Max tokens per sequence during attention capture (only for capture forward).")

    ap.add_argument("--prompt_deep_dive_ids", type=str, default=None,
                    help="Comma-separated prompt_count IDs for deep dives (e.g., '1,17,42').")
    ap.add_argument("--n_paraphrases_deep_dive", type=int, default=8)
    ap.add_argument("--include_original_deep_dive", type=str, default="false", choices=["true","false"])
    ap.add_argument("--select_heads_topk", type=int, default=6)
    ap.add_argument("--max_tokens", type=int, default=96)

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("attn_deltas")

    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    log.info("Loading SECOND prompts from %s", args.instructions_json)
    items = load_instruction_json(args.instructions_json)
    log.info("Loaded %d items.", len(items))

    keys: Optional[List[str]] = None
    if args.keys:
        keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    keys_regex = args.keys_regex

    device = args.device
    log.info("Loading models (base + ft)")
    base, ft, tokenizer = build_model_and_tokenizer(
        base_model_name_or_path=args.base_model_name_or_path,
        ft_model_name_or_path=args.ft_model_name_or_path,
        ft_lora_adapter=args.ft_lora_adapter,
        device=device,
        attn_impl=args.attn_impl,
    )
    log.info("Models ready. Layer index = %d", args.layer_index)
    log.info("B1_attn_deltas_suite initial setup OK")

    log.info("Running GLOBAL attention similarity/delta analysis (max_items=%d, batch=%d, max_attn_len=%d)...",
             args.max_items_global, args.batch_size_attn, args.max_attn_len)
    gdir = ensure_dir(outdir / "global")
    df = run_global_attention_similarity(
        items, base, ft, tokenizer, args.layer_index, device, gdir,
        prompt_span=args.prompt_span, keys=keys, keys_regex=keys_regex,
        max_items=args.max_items_global, batch_size_attn=args.batch_size_attn,
        metric_kl=True, log=log, max_attn_len=args.max_attn_len
    )
    if df.empty:
        log.warning("Global attention similarity returned empty dataframe. Check your setup/filters.")
    else:
        log.info("Global attention similarity wrote CSV/plots to: %s", gdir)

    if args.prompt_deep_dive_ids:
        dd_ids = [int(x) for x in args.prompt_deep_dive_ids.split(",") if x.strip()]
        log.info("Running prompt-level deep dives for: %s", dd_ids)
        ddir = ensure_dir(outdir / "deep_dives")
        run_prompt_deep_dives(
            items, base, ft, tokenizer, args.layer_index, device, ddir,
            prompt_span=args.prompt_span, keys=keys, keys_regex=keys_regex,
            prompt_ids=dd_ids, n_paraphrases=args.n_paraphrases_deep_dive,
            include_original=(args.include_original_deep_dive == "true"),
            select_heads_topk=args.select_heads_topk, max_tokens=args.max_tokens
        )
        readme = []
        readme.append("# Attention deltas — prompt-level deep dives")
        readme.append("")
        readme.append(f"Prompts analyzed: {', '.join(map(str, dd_ids))}")
        readme.append("Each folder `prompt_{ID}` contains:")
        readme.append("- `head_rank_by_change.csv` (heads sorted by change vs BASE)")
        readme.append("- `head_stats.csv` (avg attention mass to first/last/diag for BASE and FT)")
        readme.append("- `head{H}_attn_grid.png` (FT−BASE attention difference per paraphrase, top-K heads)")
        readme.append("- `head{H}_top_targets_across_paraphrases.csv`")
        (ddir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    r = []
    r.append("# Attention deltas suite — outputs")
    r.append("")
    r.append("## 1) Global")
    r.append("- CSV: `global/attn_similarity_layer{L}.csv`".replace("{L}", str(args.layer_index)))
    r.append("- Plots:")
    r.append(f"  - `global/attn_similarity_layer{args.layer_index}_cosine_per_head.png`")
    r.append(f"  - `global/attn_similarity_layer{args.layer_index}_cosine_heatmap.png`")
    r.append(f"  - `global/attn_similarity_layer{args.layer_index}_skl_per_head.png`")
    r.append(f"  - `global/attn_similarity_layer{args.layer_index}_skl_heatmap.png`")
    r.append(f"  - `global/attn_entropy_overlay_layer{args.layer_index}.png`")
    r.append(f"  - `global/attn_entropy_delta_layer{args.layer_index}.png`")
    r.append(f"  - `global/attn_span_overlay_layer{args.layer_index}.png`")
    r.append(f"  - `global/attn_span_delta_layer{args.layer_index}.png`")
    r.append(f"  - `global/attn_mass_input_overlay_layer{args.layer_index}.png`")
    r.append(f"  - `global/attn_mass_input_delta_layer{args.layer_index}.png`")
    if args.prompt_deep_dive_ids:
        r.append("")
        r.append("## 2) Prompt-level deep dives")
        r.append("- See `deep_dives/` directory; one folder per prompt.")
    (outdir / "README.md").write_text("\n".join(r), encoding="utf-8")

    log.info("Done. Outputs written to: %s", outdir)

if __name__ == "__main__":
    main()

