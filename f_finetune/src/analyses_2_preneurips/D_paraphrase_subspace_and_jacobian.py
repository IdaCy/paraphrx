#!/usr/bin/env python3
"""
SCRIPT D (fixed with optional batching/caching; defaults preserve original outputs)

paraphrase_subspace_and_jacobian.py

Purpose:
- For the **FIRST** dataset selection (from prep_first_dataset_sampler.py: `jacobian_prompts.jsonl`),
  probe what the **layer-L MLP** learned by estimating its directional sensitivity (Jacobian norms)
  along three families of directions:
    (1) Mean direction (normalize(mean(h))).   [semantic core]
    (2) Paraphrase-variance directions:
        - `--jacobian_mode pca`: top-K PCs of centered paraphrase activations.
        - `--jacobian_mode raw`: normalized centered vectors themselves (up to --max_directions).
    (3) Random directions (control).
  Compare **BASE** vs **FT** (and optionally multiple FT runs by re-running with different `--tag`).

- Also compute and compare the **norm of the MLP output** ||MLP(h)||₂ distributions (BASE vs FT).

Kept from your tuned setup:
- Comprehensive logging, seed=42 default.
- Robust model loading & LoRA merging; device & dtype handling.
- Layer index flag; pooling over prompt tokens (mean by default).
- Strong separation of concerns: no unrelated extras; outputs are CSVs and small, focused plots.

Outputs (to --outdir):
- Per model tag:
  * `{TAG}_paraphrase_subspace_jacobian_layer{L}.csv`   (per-prompt results)
  * `{TAG}_mlp_out_norm_stats_layer{L}.csv`             (per-prompt ||MLP(h)||₂ stats)
  * `{TAG}_pca_stats_layer{L}.csv` (if mode=pca): per-prompt explained variance ratios
- Aggregated plots:
  * `jacobian_var_bars_layer{L}.png`          (VAR across models, mean±CI)
  * `jacobian_rand_bars_layer{L}.png`         (RANDOM across models, mean±CI)
  * `jacobian_mean_dir_bars_layer{L}.png`     (MEAN across models, mean±CI)
  * `mlp_out_norm_mean_bars_layer{L}.png`     (BASE vs FT mean ||MLP(h)||₂)
  * `pca_scree_mean_layer{L}.png`             (if mode=pca; mean EVR bars)
- NEW comparative figures (single-figure overlays & deeper comparisons):
  * `jacobian_combined_layer{L}.png`                    (MEAN/VAR/RAND in one plot, colored, BASE/FT)
  * `jacobian_ratio_ft_over_base_layer{L}.png`          (FT/BASE ratios with 95% CI, per family)
  * `jacobian_diff_hist_layer{L}.png`                   (overlaid histograms of FT−BASE for MEAN & VAR)
  * `jacobian_scatter_VAR_layer{L}.png`                 (scatter BASE vs FT for VAR with y=x line)
  * `jacobian_var_box_layer{L}.png`                     (boxplots of VAR per tag, same figure)
  * `mlp_out_norm_combined_bars_layer{L}.png`           (BASE & FT norms in same plot)
  * `pca_scree_overlay_layer{L}.png`                    (if mode=pca; EVR overlay BASE vs FT)
  * `pca_cum_evr_overlay_layer{L}.png`                  (if mode=pca; cumulative EVR overlay)

Key fixes added (all **optional** and **OFF by default** to guarantee identical outputs unless enabled):
- **Batched forwards** for pre-MLP activations to speed up token processing (`--enable_batching 1`).
- **Batched Jacobian finite differences** for multiple directions at once (`--enable_batching 1`).
- **Tokenization cache** so BASE/FT share encodings (`--enable_token_cache 1`).
- **Reuse cached pre-MLP pooled activations for NL stats** to avoid re-forwarding (`--reuse_cached_h 1`).
- **Progress logs** inside main & NL loops (no effect on results).

All numerical computations & outputs are unchanged when these flags are left at default (0).


python f_finetune/src/analyses_2_preneurips/paraphrase_subspace_and_jacobian.py \
  --selection_jsonl f_finetune/data/first_dataset_sampler/jacobian_prompts.jsonl \
  --base_model_name_or_path f_finetune/model \
  --ft_model_name_or_path f_finetune/outputs_great_nolap/cp_ft_spec6layer/final \
  --layer_index 6 \
  --mlp_submap upstream \
  --upstream_variant up_proj \
  --jacobian_mode pca --topk_pca 8 \
  --eps_jacobian 0.001 --eps_list "0.0005,0.001,0.002" \
  --max_prompts 100 \
  --outdir f_finetune/outputs_great_nolap/D1_PCA_UPSTREAM_1 \
    --enable_batching 1 \
    --enable_token_cache 1 \
    --reuse_cached_h 1
"""
from __future__ import annotations
import argparse, json, logging, math, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v = v.to(torch.float32)
    n = v.norm(p=2)
    return v / (n + eps)

@dataclass
class PromptSet:
    prompt_count: int
    instruction_original: str
    input_text: str
    paraphrases: List[Tuple[str,str]]

def load_selection_jsonl(path: str | Path, max_prompts: Optional[int] = None) -> List[PromptSet]:
    items = []
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
    if max_prompts is not None:
        items = items[:max_prompts]
    return items

def build_prompt_text(instruction: str, input_text: str) -> str:
    if input_text and input_text.strip():
        return f"{instruction}\\n\\nInput: {input_text}"
    return instruction

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
        if self.mlp is None:
            raise TypeError("Could not access MLP submodule.")

class MLPInspector:
    """
    Introspects the MLP at a given layer and exposes upstream/postact/downstream maps.
    Supports SwiGLU (gate_proj, up_proj, down_proj) and GeLU (wi/wo or fc_in/fc_out).
    """
    def __init__(self, model: nn.Module, layer_index: int):
        ba = BlockAccessor(model, layer_index)
        self.mlp = ba.mlp
        self.device = next(self.mlp.parameters()).device
        self.dtype  = next(self.mlp.parameters()).dtype
        # Detect flavor
        self.has_gate = hasattr(self.mlp, "gate_proj")
        self.has_up   = hasattr(self.mlp, "up_proj")
        self.has_down = hasattr(self.mlp, "down_proj")
        self.gelu_in  = hasattr(self.mlp, "wi") or hasattr(self.mlp, "fc_in") or hasattr(self.mlp, "dense_h_to_4h")
        self.gelu_out = hasattr(self.mlp, "wo") or hasattr(self.mlp, "fc_out") or hasattr(self.mlp, "dense_4h_to_h")
        self.flavor = "swiGLU" if (self.has_up and self.has_down) else "gelu"
    def upstream_apply(self, h: torch.Tensor, variant: str = "up_proj") -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.flavor == "swiGLU":
            if variant == "up_proj":
                return self.mlp.up_proj(h)
            elif variant == "gate_proj":
                return self.mlp.gate_proj(h)
            else:
                raise ValueError("variant must be 'up_proj' or 'gate_proj'")
        else:
            # gelu flavor upstream linear
            if hasattr(self.mlp, "wi"):
                return self.mlp.wi(h)
            elif hasattr(self.mlp, "fc_in"):
                return self.mlp.fc_in(h)
            elif hasattr(self.mlp, "dense_h_to_4h"):
                return self.mlp.dense_h_to_4h(h)
            else:
                raise RuntimeError("Unknown GeLU upstream map.")
    def postact(self, h: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device, self.dtype)
        if self.flavor == "swiGLU":
            up = self.mlp.up_proj(h)
            gate = self.mlp.gate_proj(h)
            return torch.nn.functional.silu(gate) * up
        else:
            # gelu
            if hasattr(self.mlp, "wi"):
                a = self.mlp.wi(h)
            elif hasattr(self.mlp, "fc_in"):
                a = self.mlp.fc_in(h)
            elif hasattr(self.mlp, "dense_h_to_4h"):
                a = self.mlp.dense_h_to_4h(h)
            else:
                raise RuntimeError("Unknown GeLU upstream map.")
            return torch.nn.functional.gelu(a)
    def downstream_apply_from_post(self, post: torch.Tensor) -> torch.Tensor:
        post = post.to(self.device, self.dtype)
        if self.flavor == "swiGLU":
            return self.mlp.down_proj(post)
        else:
            if hasattr(self.mlp, "wo"):
                return self.mlp.wo(post)
            elif hasattr(self.mlp, "fc_out"):
                return self.mlp.fc_out(post)
            elif hasattr(self.mlp, "dense_4h_to_h"):
                return self.mlp.dense_4h_to_h(post)
            else:
                raise RuntimeError("Unknown GeLU downstream map.")
    def gate_pre(self, h: torch.Tensor) -> Optional[torch.Tensor]:
        if self.flavor != "swiGLU":
            return None
        h = h.to(self.device, self.dtype)
        return self.mlp.gate_proj(h)

def build_model_and_tokenizer(
    base_model_name_or_path: str,
    ft_model_name_or_path: Optional[str],
    ft_lora_adapter: Optional[str],
    device: str,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    base = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device).eval()

    if ft_model_name_or_path is None and ft_lora_adapter is None:
        raise ValueError("Provide --ft_model_name_or_path (merged FT) OR --ft_lora_adapter (LoRA adapter dir).")

    if ft_lora_adapter is not None:
        apath = Path(ft_lora_adapter)
        if apath.exists() and (apath / "adapter_config.json").exists():
            if not _HAS_PEFT:
                raise RuntimeError("peft not installed but --ft_lora_adapter was provided.")
            ft = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
            from peft import PeftModel  # lazy import to avoid hard dep if not used
            ft = PeftModel.from_pretrained(ft, str(apath))
            ft = ft.merge_and_unload().eval()
            return base, ft, tokenizer
        if apath.exists() and (apath / "config.json").exists():
            ft = AutoModelForCausalLM.from_pretrained(str(apath), torch_dtype=dtype).to(device).eval()
            return base, ft, tokenizer

    if ft_model_name_or_path is None:
        raise ValueError("When --ft_lora_adapter is not a local adapter dir, you must provide --ft_model_name_or_path.")
    ft = AutoModelForCausalLM.from_pretrained(ft_model_name_or_path, torch_dtype=dtype).to(device).eval()
    return base, ft, tokenizer

@dataclass
class Encoded:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_mask: torch.Tensor
    tokens: List[str]

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
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    return Encoded(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        prompt_mask=prompt_mask.to(device),
        tokens=tokens
    )

def capture_pre_mlp_activations(model: nn.Module, layer_index: int, enc: Encoded) -> torch.Tensor:
    ba = BlockAccessor(model, layer_index)
    captured = {"x": None}
    def pre_hook(module, inputs):
        x = inputs[0]
        captured["x"] = x.detach().to("cpu")
    h = ba.mlp.register_forward_pre_hook(pre_hook, with_kwargs=False)
    with torch.inference_mode():
        _ = model(input_ids=enc.input_ids.unsqueeze(0), attention_mask=enc.attention_mask.unsqueeze(0))
    h.remove()
    x = captured["x"]
    if x is None:
        raise RuntimeError("Failed to capture pre-MLP activations.")
    T = x.shape[1]
    return x[0, :T, :].contiguous()

# ---------- NEW: Batched capture (optional) ----------
def capture_pre_mlp_activations_batch(model: nn.Module, layer_index: int, enc_list: List[Encoded]) -> torch.Tensor:
    """
    Returns a CPU tensor of shape [B, T_max, d_model] with padding where necessary.
    Uses the same pre-hook mechanism but feeds a batch.
    """
    if len(enc_list) == 0:
        raise ValueError("enc_list must be non-empty")
    ba = BlockAccessor(model, layer_index)
    captured = {"x": None}
    def pre_hook(module, inputs):
        captured["x"] = inputs[0].detach().to("cpu")  # [B, T, d]
    h = ba.mlp.register_forward_pre_hook(pre_hook, with_kwargs=False)
    with torch.inference_mode():
        # Pad to batch
        pad_id = getattr(getattr(model, "config", object()), "pad_token_id", 0) or 0
        input_ids = torch.nn.utils.rnn.pad_sequence([e.input_ids for e in enc_list], batch_first=True, padding_value=pad_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence([e.attention_mask for e in enc_list], batch_first=True, padding_value=0)
        device = enc_list[0].input_ids.device
        _ = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
    h.remove()
    X = captured["x"]
    if X is None:
        raise RuntimeError("Failed to capture pre-MLP activations (batched).")
    return X  # [B, T, d] on CPU

def get_mlp_function(model: nn.Module, layer_index: int):
    ba = BlockAccessor(model, layer_index)
    mlp = ba.mlp
    mlp_device = next(mlp.parameters()).device
    mlp_dtype  = next(mlp.parameters()).dtype
    def f(h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 1:
            h = h.unsqueeze(0)
        x = h.to(device=mlp_device, dtype=mlp_dtype).unsqueeze(0)
        with torch.inference_mode():
            y = mlp(x)
        return y[0,0,:].to("cpu", dtype=torch.float32)
    return f

# ---------- NEW: Batched MLP prober (optional wrapper that preserves numerical order) ----------
def get_mlp_prober(model: nn.Module, layer_index: int, mlp_submap: str = "full", upstream_variant: str = "up_proj"):
    """
    Returns a function f(h)->y according to selected submap.
    - full: calls the model's MLP block (original behavior).
    - upstream: applies upstream linear only (expects residual space h).
    - downstream: applies downstream linear to post-activation φ(W h) (expects residual space h).
    """
    if mlp_submap == "full":
        return get_mlp_function(model, layer_index)
    insp = MLPInspector(model, layer_index)
    def f(h: torch.Tensor) -> torch.Tensor:
        x = h.unsqueeze(0).unsqueeze(0) if h.dim()==1 else h
        if mlp_submap == "upstream":
            y = insp.upstream_apply(x, variant=upstream_variant)
        elif mlp_submap == "downstream":
            post = insp.postact(x)
            y = insp.downstream_apply_from_post(post)
        else:
            raise ValueError("mlp_submap invalid")
        y = y.to("cpu", dtype=torch.float32)
        if y.dim()==3: y = y[0,0,:]
        return y
    return f

# Batched variant that maintains identical math by iterating under the hood (but enables FD batching)
def get_mlp_prober_batched(model: nn.Module, layer_index: int, mlp_submap: str = "full", upstream_variant: str = "up_proj"):
    f_single = get_mlp_prober(model, layer_index, mlp_submap, upstream_variant)
    def f_batch(H: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(H.shape[0]):
            outs.append(f_single(H[i]))
        return torch.stack(outs, dim=0)
    return f_batch

@dataclass
class EncodedBatch:
    Htok: torch.Tensor             # [B, T, d] CPU
    prompt_masks: List[torch.Tensor]
    pooled: torch.Tensor           # [B, d] CPU (float32)

def pool_tokens(H: torch.Tensor, mask: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    H = H.to(torch.float32)
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        idx = torch.arange(H.shape[0], device=H.device)
    if mode == "mean":
        return H[idx].mean(0)
    elif mode == "first":
        return H[int(idx[0].item())]
    else:
        return H[int(idx[-1].item())]

# ---------- NEW: Batched pooling helper (optional) ----------
def pool_tokens_batch(H: torch.Tensor, prompt_masks: List[torch.Tensor], mode: str="mean") -> torch.Tensor:
    outs = []
    for i, m in enumerate(prompt_masks):
        idx = m.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            outs.append(H[i].mean(0))
        elif mode == "mean":
            outs.append(H[i, idx].mean(0))
        elif mode == "first":
            outs.append(H[i, int(idx[0].item())])
        else:
            outs.append(H[i, int(idx[-1].item())])
    return torch.stack(outs, dim=0).to(torch.float32)  # [B, d]

def estimate_jacobian_norm(f, center: torch.Tensor, direction: torch.Tensor, eps: float = 1e-3) -> float:
    c = center.to(torch.float32)
    d = normalize(direction)
    y1 = f(c + eps * d)
    y2 = f(c - eps * d)
    g = (y1 - y2) / (2.0 * eps)
    return float(g.norm(p=2).item())

# ---------- NEW: Batched FD Jacobian for many directions at once (optional) ----------
def jacobian_norms_batched(f_batch, center: torch.Tensor, dirs: List[torch.Tensor], eps_list: List[float]) -> List[float]:
    if len(dirs) == 0:
        return []
    C = center.unsqueeze(0).repeat(len(dirs), 1).to(torch.float32)
    D = torch.stack([normalize(d) for d in dirs], dim=0).to(torch.float32)  # [K, d]
    vals_per_eps = []
    for eps in eps_list:
        Yp = f_batch(C + eps * D)  # [K, d_out]
        Ym = f_batch(C - eps * D)  # [K, d_out]
        G = (Yp - Ym) / (2.0 * eps)  # [K, d_out]
        #vals_per_eps.append(torch.linalg.vector_norm(G, ord=2, dim=1).cpu().numpy())
        vals_per_eps.append(torch.linalg.vector_norm(G, ord=2, dim=1).detach().cpu().numpy())
    return list(np.mean(np.stack(vals_per_eps, axis=0), axis=0))

def main():
    ap = argparse.ArgumentParser(description="Paraphrase subspace + Jacobian probes for MLP")
    ap.add_argument("--selection_jsonl", type=str, required=True, help="jacobian_prompts.jsonl from sampler")
    ap.add_argument("--base_model_name_or_path", type=str, required=True)
    ap.add_argument("--ft_model_name_or_path", type=str, default=None)
    ap.add_argument("--ft_lora_adapter", type=str, default=None)
    ap.add_argument("--tag", type=str, default="FT")
    ap.add_argument("--layer_index", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--prompt_span", type=str, default="no_bos", choices=["no_bos","pre_eos","all"])
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean","first","last"])
    ap.add_argument("--jacobian_mode", type=str, default="pca", choices=["pca","raw"])
    ap.add_argument("--topk_pca", type=int, default=8)
    ap.add_argument("--max_directions", type=int, default=50)
    ap.add_argument("--eps_jacobian", type=float, default=1e-3)

    ap.add_argument("--eps_list", type=str, default="", help="Comma-separated epsilons to sweep in addition to --eps_jacobian")
    ap.add_argument("--mlp_submap", type=str, default="full", choices=["full","upstream","downstream"], help="Which MLP sub-map to probe with Jacobians")
    ap.add_argument("--upstream_variant", type=str, default="up_proj", choices=["up_proj","gate_proj"], help="When mlp_submap=upstream and SwiGLU, choose linear")
    ap.add_argument("--space", type=str, default="auto", choices=["auto","premlp","postact"], help="Activation space to capture for NL experiment")
    ap.add_argument("--sparsity_tau", type=float, default=1e-3, help="Absolute threshold for post-activation sparsity")
    ap.add_argument("--pca_k_dir", type=int, default=16, help="Top-K PCs for directional compression in NL experiment")
    ap.add_argument("--n_random_dirs", type=int, default=8)

    # optional speed flags
    ap.add_argument("--enable_batching", type=int, default=0, help="Enable batched forwards & FD-Jacobian (0/1). Default 0 preserves original behavior.")
    ap.add_argument("--enable_token_cache", type=int, default=0, help="Cache tokenizations across models (0/1).")
    ap.add_argument("--reuse_cached_h", type=int, default=0, help="Reuse cached pooled pre-MLP activations for NL stats (0/1).")

    args = ap.parse_args()

    #cfg_bits = [f"L{args.layer_index}", args.mlp_submap, args.jacobian_mode]
    #if args.mlp_submap == "upstream":
    #    cfg_bits.append(args.upstream_variant)
    #plot_suffix = "_" + "_".join(cfg_bits)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("jacobian")

    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    sets = load_selection_jsonl(args.selection_jsonl, max_prompts=args.max_prompts)
    log.info("Loaded %d prompt sets for Jacobian.", len(sets))
    base, ft, tokenizer = build_model_and_tokenizer(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, args.device
    )

    per_model_rows: Dict[str, List[dict]] = {"BASE": [], f"FT_{args.tag}": []}
    pca_rows: Dict[str, List[dict]] = {"BASE": [], f"FT_{args.tag}": []}
    mlp_norm_rows: Dict[str, List[dict]] = {"BASE": [], f"FT_{args.tag}": []}

    token_cache: Dict[str, Encoded] = {}
    cached_H_by_tag: Dict[str, Dict[int, torch.Tensor]] = {"BASE": {}, f"FT_{args.tag}": {}}

    def get_encoded(text: str) -> Encoded:
        if args.enable_token_cache:
            if text in token_cache:
                return token_cache[text]
            e = encode_prompt(tokenizer, text, args.device, args.prompt_span)
            token_cache[text] = e
            return e
        else:
            return encode_prompt(tokenizer, text, args.device, args.prompt_span)

    def pool_tokens(H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        H = H.to(torch.float32)
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            idx = torch.arange(H.shape[0], device=H.device)
        if args.pooling == "mean":
            return H[idx].mean(0)
        elif args.pooling == "first":
            return H[int(idx[0].item())]
        else:
            return H[int(idx[-1].item())]

    def run_for_model(tag: str, model: nn.Module):
        mlp_f = get_mlp_prober(model, args.layer_index, args.mlp_submap, args.upstream_variant)
        mlp_f_batch = get_mlp_prober_batched(model, args.layer_index, args.mlp_submap, args.upstream_variant)

        for idx, ps in enumerate(sets, 1):
            if (idx % 5) == 0 or idx == 1:
                log.info("[%s] Processing prompt %d/%d (prompt_count=%s)", tag, idx, len(sets), ps.prompt_count)

            # Encode all paraphrases for this prompt
            texts = [build_prompt_text(txt, ps.input_text) for _, txt in ps.paraphrases]
            if len(texts) < 3:
                log.warning("prompt_count=%s has <3 paraphrases; skipping.", ps.prompt_count)
                continue

            if args.enable_batching:
                encs = [get_encoded(t) for t in texts]
                Htok = capture_pre_mlp_activations_batch(model, args.layer_index, encs)  # [B, T, d] CPU
                pooled = pool_tokens_batch(Htok, [e.prompt_mask.cpu() for e in encs], args.pooling)  # [B, d]
                Hmat = pooled  # [n, d] CPU float32
            else:
                hs: List[torch.Tensor] = []
                for t in texts:
                    enc = get_encoded(t)
                    H = capture_pre_mlp_activations(model, args.layer_index, enc)
                    h = pool_tokens(H, enc.prompt_mask.cpu())
                    hs.append(h)
                Hmat = torch.stack(hs, dim=0)  # [n, d]

            # Cache pooled H for potential NL reuse
            if args.reuse_cached_h:
                cached_H_by_tag[tag][ps.prompt_count] = Hmat.clone()

            mean_h = Hmat.mean(0)
            V = Hmat - mean_h.unsqueeze(0)

            # MLP output norms
            norms = []
            if args.enable_batching:
                y = mlp_f_batch(Hmat)  # [n, d_out]
                #norms = torch.linalg.vector_norm(y, ord=2, dim=1).cpu().numpy().tolist()
                norms = torch.linalg.vector_norm(y, ord=2, dim=1).detach().cpu().numpy().tolist()
            else:
                for i in range(Hmat.shape[0]):
                    y = mlp_f(Hmat[i])
                    norms.append(float(y.norm(p=2).item()))
            mlp_norm_rows[tag].append({
                "prompt_count": ps.prompt_count,
                "n": int(Hmat.shape[0]),
                "mlp_out_norm_mean": float(np.mean(norms)),
                "mlp_out_norm_std": float(np.std(norms, ddof=1) if len(norms)>1 else 0.0),
                "mlp_out_norm_min": float(np.min(norms)),
                "mlp_out_norm_max": float(np.max(norms)),
            })

            dirs_mean = [normalize(mean_h)]
            dirs_random = []
            D = Hmat.shape[1]
            for _ in range(args.n_random_dirs):
                r = torch.randn(D, dtype=torch.float32)
                dirs_random.append(normalize(r))

            if args.jacobian_mode == "pca":
                Vc = (V - V.mean(0, keepdim=True)).to(torch.float32)
                U, S, VT = torch.linalg.svd(Vc, full_matrices=False)
                comps = VT[:args.topk_pca, :]
                dirs_var = [normalize(comps[k]) for k in range(comps.shape[0])]
                var = (S**2)
                total = var.sum().item() + 1e-8
                ratios = (var / total).cpu().numpy()
                row = {"prompt_count": ps.prompt_count, "evr_total_topk": float(ratios[:args.topk_pca].sum())}
                for i in range(min(len(ratios), args.topk_pca)):
                    row[f"evr_{i+1}"] = float(ratios[i])
                pca_rows[tag].append(row)
            else:
                idxs = np.random.permutation(V.shape[0])[:args.max_directions]
                dirs_var = [normalize(V[i]) for i in idxs]

            eps_list = [float(args.eps_jacobian)]
            if args.eps_list:
                eps_list += [float(e.strip()) for e in args.eps_list.split(",") if e.strip()]

            def jac_for_dirs(dirs):
                if args.enable_batching:
                    return jacobian_norms_batched(mlp_f_batch, mean_h, dirs, eps_list)
                else:
                    vals_all = []
                    for eps in eps_list:
                        vals = [estimate_jacobian_norm(mlp_f, mean_h, d, eps) for d in dirs]
                        vals_all.append(vals)
                    return list(np.mean(np.array(vals_all), axis=0))

            jac_mean_vals = jac_for_dirs(dirs_mean)
            jac_var_vals  = jac_for_dirs(dirs_var)
            jac_rand_vals = jac_for_dirs(dirs_random)
            per_model_rows[tag].append({
                "prompt_count": ps.prompt_count,
                "n": int(Hmat.shape[0]),
                "jac_MEAN": float(np.mean(jac_mean_vals)),
                "jac_VAR_mean": float(np.mean(jac_var_vals)) if len(jac_var_vals)>0 else np.nan,
                "jac_VAR_std": float(np.std(jac_var_vals, ddof=1)) if len(jac_var_vals)>1 else 0.0,
                "jac_RAND_mean": float(np.mean(jac_rand_vals)),
                "jac_RAND_std": float(np.std(jac_rand_vals, ddof=1)) if len(jac_rand_vals)>1 else 0.0,
                "mode": args.jacobian_mode,
            })

    logging.getLogger("jacobian").info("Running Jacobian probes for BASE...")
    run_for_model("BASE", base)
    logging.getLogger("jacobian").info("Running Jacobian probes for FT (%s)...", args.tag)
    run_for_model(f"FT_{args.tag}", ft)

    # --- Persist CSVs (unchanged) ---
    for tag, rows in per_model_rows.items():
        pd.DataFrame(rows).to_csv(outdir / f"{tag}_paraphrase_subspace_jacobian_layer{args.layer_index}.csv", index=False)
    for tag, rows in mlp_norm_rows.items():
        pd.DataFrame(rows).to_csv(outdir / f"{tag}_mlp_out_norm_stats_layer{args.layer_index}.csv", index=False)
    if args.jacobian_mode == "pca":
        for tag, rows in pca_rows.items():
            pd.DataFrame(rows).to_csv(outdir / f"{tag}_pca_stats_layer{args.layer_index}.csv", index=False)

    # --- Existing mean±CI bar helpers (kept) ---
    def agg_bar(data: Dict[str, List[dict]], metric: str, title: str, fname: str):
        tags = list(data.keys())
        means = []
        cis = []
        for tag in tags:
            df = pd.DataFrame(data[tag])
            vals = df[metric].dropna().values
            if len(vals) == 0:
                means.append(0.0); cis.append(0.0); continue
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
            ci = 1.96 * s / math.sqrt(len(vals)) if len(vals)>1 else 0.0
            means.append(m); cis.append(ci)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(tags, means, yerr=cis)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=170)
        plt.close(fig)

    # Original single-metric figures
    agg_bar(per_model_rows, "jac_VAR_mean",  f"Jacobian (VAR — mode={args.jacobian_mode}) — layer {args.layer_index}", f"jacobian_var_bars_layer{args.layer_index}.png")
    agg_bar(per_model_rows, "jac_RAND_mean", f"Jacobian (RANDOM) — layer {args.layer_index}", f"jacobian_rand_bars_layer{args.layer_index}.png")
    agg_bar(per_model_rows, "jac_MEAN",      f"Jacobian (MEAN dir) — layer {args.layer_index}", f"jacobian_mean_dir_bars_layer{args.layer_index}.png")

    tags = list(mlp_norm_rows.keys())
    means = []; cis = []
    for tag in tags:
        df = pd.DataFrame(mlp_norm_rows[tag])
        vals = df["mlp_out_norm_mean"].dropna().values
        m = float(np.mean(vals)) if len(vals)>0 else 0.0
        s = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
        ci = 1.96 * s / math.sqrt(len(vals)) if len(vals)>1 else 0.0
        means.append(m); cis.append(ci)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(tags, means, yerr=cis)
    ax.set_title(f"||MLP(h)||₂ (mean over paraphrases) — layer {args.layer_index}")
    fig.tight_layout()
    #fig.savefig(outdir / f"mlp_out_norm_mean_bars_layer{args.layer_index}.png", dpi=170)
    #ax.set_title(f"Jacobian (VAR — mode={args.jacobian_mode}) — {args.mlp_submap}{('/'+args.upstream_variant) if args.mlp_submap=='upstream' else ''} — layer {args.layer_index}")
    #fig.savefig(outdir / f"jacobian_var_bars_layer{args.layer_index}{plot_suffix}.png", dpi=170)
    plt.close(fig)

    if args.jacobian_mode == "pca":
        evr_cols = [f"evr_{i+1}" for i in range(args.topk_pca)]
        evr_means = []; evr_cis = []
        base_pca = pd.DataFrame(pca_rows["BASE"])
        if not base_pca.empty:
            for c in evr_cols:
                vals = base_pca[c].dropna().values
                m = float(np.mean(vals)) if len(vals)>0 else 0.0
                s = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
                ci = 1.96 * s / math.sqrt(len(vals)) if len(vals)>1 else 0.0
                evr_means.append(m); evr_cis.append(ci)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar([f"PC{i+1}" for i in range(len(evr_means))], evr_means, yerr=evr_cis)
            ax.set_title(f"PCA explained variance ratio — mean across prompts (layer {args.layer_index})")
            fig.tight_layout()
            fig.savefig(outdir / f"pca_scree_mean_layer{args.layer_index}.png", dpi=170)
            plt.close(fig)

    
    # ----------------------
    # Nonlinearity experiment: pre-MLP -> post-activation compression and gate diagnostics
    # We compute per-prompt stats in the SAME run loop above; aggregate and plot here.
    # ----------------------
    try:
        # We recompute per-prompt stats using cached Hmat for each model tag by re-running a light pass
        # to avoid changing existing data structures.
        def nl_stats_for_model(tag: str, model: nn.Module):
            insp = MLPInspector(model, args.layer_index)
            rows = []
            for idx, ps in enumerate(sets, 1):
                if (idx % 5) == 0 or idx == 1:
                    logging.getLogger("jacobian").info("[NL %s] Prompt %d/%d (prompt_count=%s)", tag, idx, len(sets), ps.prompt_count)

                # Either reuse cached pooled H or forward again (original behavior)
                if args.reuse_cached_h and (ps.prompt_count in cached_H_by_tag[tag]):
                    H = cached_H_by_tag[tag][ps.prompt_count].to(torch.float32)  # [n, d]
                else:
                    # Original behavior: capture again (optionally batched)
                    texts = [build_prompt_text(txt, ps.input_text) for _, txt in ps.paraphrases]
                    hs_list = []
                    if args.enable_batching:
                        encs = [encode_prompt(tokenizer, t, args.device, args.prompt_span) for t in texts]
                        Htok = capture_pre_mlp_activations_batch(model, args.layer_index, encs)
                        pooled = pool_tokens_batch(Htok, [e.prompt_mask.cpu() for e in encs], args.pooling)
                        H = pooled
                    else:
                        for t in texts:
                            enc = encode_prompt(tokenizer, t, args.device, args.prompt_span)
                            Htok = capture_pre_mlp_activations(model, args.layer_index, enc)
                            h = pool_tokens(Htok, enc.prompt_mask.cpu())
                            hs_list.append(h.to(torch.float32))
                        if len(hs_list) < 3:
                            continue
                        H = torch.stack(hs_list, dim=0)  # [n, d_model]

                if H.shape[0] < 3:
                    continue

                # pre-activation and post-activation in ff dim
                with torch.inference_mode():
                    # compute upstream projections
                    if insp.flavor == "swiGLU":
                        up = insp.upstream_apply(H, variant="up_proj")
                        gate = insp.upstream_apply(H, variant="gate_proj")
                        post = torch.nn.functional.silu(gate) * up
                        preact = up  # choose up as the target for PCA variance comparison
                        gate_pre = gate
                    else:
                        if hasattr(insp.mlp, "wi"):
                            a = insp.mlp.wi(H.to(insp.device, insp.dtype))
                        elif hasattr(insp.mlp, "fc_in"):
                            a = insp.mlp.fc_in(H.to(insp.device, insp.dtype))
                        elif hasattr(insp.mlp, "dense_h_to_4h"):
                            a = insp.mlp.dense_h_to_4h(H.to(insp.device, insp.dtype))
                        else:
                            raise RuntimeError("Unknown GeLU upstream map.")
                        preact = a
                        post = torch.nn.functional.gelu(a)
                        gate_pre = a  # for GeLU, sign<0 plays similar role
                    preact = preact.to(torch.float32).cpu()
                    post = post.to(torch.float32).cpu()
                    Hcpu = H.cpu()

                def stats_matrix(X: torch.Tensor):
                    X = X.to(torch.float32)
                    mu = X.mean(0, keepdim=True)
                    Xc = X - mu
                    cov = (Xc.T @ Xc) / max(1, X.shape[0]-1)
                    tr = float(torch.trace(cov).item())
                    dim = X.shape[1]
                    # top-k eigs
                    k = min(args.pca_k_dir, dim)
                    try:
                        evals = torch.linalg.eigvalsh(cov)[-k:].flip(0)
                        evals = evals.to(torch.float32).cpu().numpy()
                    except Exception:
                        # fall back to SVD
                        U,S,VT = torch.linalg.svd(Xc, full_matrices=False)
                        evals = (S**2).to(torch.float32).cpu().numpy()[:k]
                    mean_norm = float(mu.norm(p=2).item())
                    pr = (tr**2) / float((cov @ cov).trace().item() + 1e-8)
                    snr = (mean_norm**2) / (tr/dim + 1e-8)
                    return tr, dim, mean_norm, pr, snr, evals

                tr_h, dh, m_h, pr_h, snr_h, _ = stats_matrix(Hcpu)
                tr_pre, dpre, m_pre, pr_pre, snr_pre, evals_pre = stats_matrix(preact)
                tr_post, dpost, m_post, pr_post, snr_post, evals_post = stats_matrix(post)

                # directional compression on preact PCs
                X = preact - preact.mean(0, keepdim=True)
                U,S,VT = torch.linalg.svd(X, full_matrices=False)
                k = min(args.pca_k_dir, VT.shape[0])
                comps = VT[:k,:].to(torch.float32)
                # variances along PCs before vs after
                cov_pre = (X.T @ X) / max(1, X.shape[0]-1)
                Y = post - post.mean(0, keepdim=True)
                cov_post = (Y.T @ Y) / max(1, Y.shape[0]-1)
                comp_vars_pre = (comps @ cov_pre @ comps.T).diagonal().cpu().numpy()
                comp_vars_post = (comps @ cov_post @ comps.T).diagonal().cpu().numpy()
                c_ratio = (comp_vars_post + 1e-12) / (comp_vars_pre + 1e-12)
                med_c = float(np.median(c_ratio))
                mean_c = float(np.mean(c_ratio))

                # gate/sparsity
                if insp.flavor == "swiGLU":
                    g = gate_pre.to(torch.float32).cpu().numpy()
                    frac_neg = float((g < 0).mean())
                    neg_margin = float(np.mean(np.maximum(0.0, -g)))
                else:
                    a = gate_pre.to(torch.float32).cpu().numpy()
                    frac_neg = float((a < 0).mean())
                    neg_margin = float(np.mean(np.maximum(0.0, -a)))
                post_abs = post.abs().cpu().numpy()
                spars_abs = float((post_abs < args.sparsity_tau).mean())
                med = np.median(post_abs)
                spars_adapt = float((post_abs < (0.01 * (med if med>0 else 1.0))).mean())

                rows.append({
                    "prompt_count": ps.prompt_count,
                    "tr_residual": tr_h, "dim_residual": dh, "snr_residual": snr_h,
                    "tr_preact": tr_pre, "dim_preact": dpre, "snr_preact": snr_pre,
                    "tr_post": tr_post, "dim_post": dpost, "snr_post": snr_post,
                    "log_compression_preact2post": float(np.log((tr_post/dpost + 1e-12)/(tr_pre/dpre + 1e-12))),
                    "raw_delta_trace": float(tr_post - tr_pre),
                    "pr_preact": pr_pre, "pr_post": pr_post,
                    "pc_mean_c_ratio": mean_c, "pc_median_c_ratio": med_c,
                    "gate_frac_neg": frac_neg, "gate_neg_margin": neg_margin,
                    "spars_post_abs": spars_abs, "spars_post_adapt": spars_adapt,
                })
            return pd.DataFrame(rows)

        nl_rows = {}
        for tag, model in [("BASE", base), (f"FT_{args.tag}", ft)]:
            logging.getLogger("jacobian").info("NL experiment stats for %s ...", tag)
            nl_rows[tag] = nl_stats_for_model(tag, model)
            nl_rows[tag].to_csv(outdir / f"{tag}_nl_stats_layer{args.layer_index}.csv", index=False)

        # Paired merge on prompt_count
        merged_nl = nl_rows["BASE"].merge(nl_rows[f"FT_{args.tag}"], on="prompt_count", suffixes=("_BASE","_FT"))

        # ΔΔ log compression
        def mean_ci(vals):
            if len(vals)==0: return 0.0,0.0
            m=float(np.mean(vals)); s=float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
            ci=1.96*s/math.sqrt(len(vals)) if len(vals)>1 else 0.0
            return m,ci

        fig, ax = plt.subplots(figsize=(7.5,4.5))
        mB, cB = mean_ci(merged_nl["log_compression_preact2post_BASE"].values)
        mF, cF = mean_ci(merged_nl["log_compression_preact2post_FT"].values)
        ax.bar(["BASE","FT"], [mB,mF], yerr=[cB,cF])
        ax.set_title(f"log compression (preact→post) — layer {args.layer_index} (lower is more compression)")
        fig.tight_layout()
        fig.savefig(outdir / f"nl_log_compression_bars_layer{args.layer_index}.png", dpi=170)
        plt.close(fig)

        # Overlay histogram of raw compression ratios
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax.hist(np.exp(merged_nl["log_compression_preact2post_BASE"].values), bins=40, alpha=0.6, label="BASE")
        ax.hist(np.exp(merged_nl["log_compression_preact2post_FT"].values), bins=40, alpha=0.6, label="FT")
        ax.set_title(f"Compression ratio preact→post (trace/dim) — overlay — L{args.layer_index}")
        ax.legend()
        fig.tight_layout(); fig.savefig(outdir / f"nl_compression_ratio_hist_overlay_layer{args.layer_index}.png", dpi=170); plt.close(fig)

        # Gate negativity & sparsity overlay bars
        for metric,label in [("gate_frac_neg","Gate frac < 0"), ("spars_post_abs","Post sparsity (abs τ)"), ("spars_post_adapt","Post sparsity (1% median)")]:
            fig, ax = plt.subplots(figsize=(7.5,4.5))
            mB,cB = mean_ci(merged_nl[f"{metric}_BASE"].values)
            mF,cF = mean_ci(merged_nl[f"{metric}_FT"].values)
            ax.bar(["BASE","FT"], [mB,mF], yerr=[cB,cF])
            ax.set_title(f"{label} — overlay — L{args.layer_index}")
            fig.tight_layout(); fig.savefig(outdir / f"nl_{metric}_bars_layer{args.layer_index}.png", dpi=170); plt.close(fig)

        # Directional compression violin (median c_ratio across PCs)
        fig, ax = plt.subplots(figsize=(8,4.5))
        ax.violinplot([merged_nl["pc_median_c_ratio_BASE"].values, merged_nl["pc_median_c_ratio_FT"].values], showmeans=True)
        ax.set_xticks([1,2]); ax.set_xticklabels(["BASE","FT"])
        ax.set_title(f"Directional compression (median over PCs) — L{args.layer_index} (lower is more)")
        fig.tight_layout(); fig.savefig(outdir / f"nl_dir_compression_violin_layer{args.layer_index}.png", dpi=170); plt.close(fig)

        # Paired scatter for log compression with y=x
        fig, ax = plt.subplots(figsize=(5.5,5.5))
        x = merged_nl["log_compression_preact2post_BASE"].values
        y = merged_nl["log_compression_preact2post_FT"].values
        ax.scatter(x,y, s=12, alpha=0.6)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, '--')
        ax.set_xlabel("BASE"); ax.set_ylabel("FT"); ax.set_title(f"log compression preact→post (paired) — L{args.layer_index}")
        fig.tight_layout(); fig.savefig(outdir / f"nl_log_compression_scatter_layer{args.layer_index}.png", dpi=170); plt.close(fig)

    except Exception as e:
        logging.getLogger("jacobian").warning("NL experiment failed or partial: %s", e)
# ======================
    # NEW comparative plots
    # ======================

    # Helper: aligned per-prompt dataframe for paired diffs/ratios
    base_df = pd.DataFrame(per_model_rows["BASE"])
    ft_df   = pd.DataFrame(per_model_rows[f"FT_{args.tag}"])
    merged  = None
    if not base_df.empty and not ft_df.empty:
        merged = base_df.merge(ft_df, on="prompt_count", suffixes=("_BASE","_FT"))

    # Paired statistical tests (BASE vs FT) for MEAN/VAR/RANDOM
    try:
        import numpy as _np
        try:
            from scipy.stats import ttest_rel, wilcoxon
            _HAS_SCIPY = True
        except Exception:
            _HAS_SCIPY = False

        stats_rows = []
        def paired_tests(bcol, fcol, label):
            xb = merged[bcol].values
            xf = merged[fcol].values
            mask = _np.isfinite(xb) & _np.isfinite(xf)
            xb = xb[mask]; xf = xf[mask]
            diffs = xf - xb
            res = {"metric": label,
                   "n": int(diffs.size),
                   "mean_base": float(xb.mean()) if xb.size else _np.nan,
                   "mean_ft": float(xf.mean()) if xf.size else _np.nan,
                   "mean_diff": float(diffs.mean()) if diffs.size else _np.nan,
                   "std_diff": float(diffs.std(ddof=1)) if diffs.size>1 else 0.0}
            if _HAS_SCIPY and diffs.size >= 2:
                res["ttest_t"], res["ttest_p"] = ttest_rel(xf, xb)
                try:
                    w = wilcoxon(diffs)
                    res["wilcoxon_stat"], res["wilcoxon_p"] = w.statistic, w.pvalue
                except Exception:
                    res["wilcoxon_stat"], res["wilcoxon_p"] = _np.nan, _np.nan
            else:
                res["ttest_t"] = res["ttest_p"] = _np.nan
                res["wilcoxon_stat"] = res["wilcoxon_p"] = _np.nan
            return res

        if merged is not None:
            stats_rows.append(paired_tests("jac_MEAN_BASE", "jac_MEAN_FT", "MEAN"))
            stats_rows.append(paired_tests("jac_VAR_mean_BASE", "jac_VAR_mean_FT", "VAR"))
            stats_rows.append(paired_tests("jac_RAND_mean_BASE", "jac_RAND_mean_FT", "RANDOM"))

            stats_df = pd.DataFrame(stats_rows)
            stats_path = outdir / f"paired_stats_layer{args.layer_index}.csv"
            stats_df.to_csv(stats_path, index=False)
            logging.getLogger("jacobian").info("Paired stats written: %s\\n%s", stats_path, stats_df.to_string(index=False))
    except Exception as e:
        logging.getLogger("jacobian").warning("Paired tests failed: %s", e)

    # 1) Combined jacobian plot in one figure: MEAN vs VAR vs RANDOM for BASE and FT
    try:
        if merged is not None and not merged.empty:
            # compute group means ± CI
            def mean_ci(vals):
                vals = np.asarray(vals, dtype=float)
                if len(vals) == 0: return 0.0, 0.0
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                ci = 1.96 * s / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
                return m, ci

            metrics = [
                ("jac_MEAN_BASE","jac_MEAN_FT","MEAN"),
                ("jac_VAR_mean_BASE","jac_VAR_mean_FT","VAR"),
                ("jac_RAND_mean_BASE","jac_RAND_mean_FT","RANDOM"),
            ]
            x = np.arange(len(metrics))
            width = 0.35

            fig, ax = plt.subplots(figsize=(9,5))
            base_means = []; base_cis = []
            ft_means   = []; ft_cis   = []
            labels = []
            for mb, mf, lab in metrics:
                mB, cB = mean_ci(merged[mb].dropna().values)
                mF, cF = mean_ci(merged[mf].dropna().values)
                base_means.append(mB); base_cis.append(cB)
                ft_means.append(mF);   ft_cis.append(cF)
                labels.append(lab)
            ax.bar(x - width/2, base_means, width, yerr=base_cis, label="BASE")
            ax.bar(x + width/2, ft_means,   width, yerr=ft_cis,   label=f"FT_{args.tag}")
            ax.set_xticks(x); ax.set_xticklabels(labels)
            ax.set_title(f"Jacobian sensitivities by direction — layer {args.layer_index}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / f"jacobian_combined_layer{args.layer_index}.png", dpi=170)
            plt.close(fig)
    except Exception as e:
        logging.warning("Failed to plot jacobian_combined: %s", e)

    # 2) Ratios FT/BASE for MEAN/VAR/RANDOM (paired per prompt), mean±CI
    try:
        if merged is not None and not merged.empty:
            ratios = {}
            for base_col, ft_col, lab in [
                ("jac_MEAN_BASE","jac_MEAN_FT","MEAN"),
                ("jac_VAR_mean_BASE","jac_VAR_mean_FT","VAR"),
                ("jac_RAND_mean_BASE","jac_RAND_mean_FT","RANDOM"),
            ]:
                # Avoid div-by-zero; drop rows with base<=0
                valid = merged[merged[base_col] > 0]
                if valid.empty:
                    ratios[lab] = np.array([])
                else:
                    ratios[lab] = (valid[ft_col].values / valid[base_col].values)

            labs = ["MEAN","VAR","RANDOM"]
            x = np.arange(len(labs)); width=0.6
            means = []; cis=[]
            for lab in labs:
                arr = ratios[lab]
                if arr.size == 0:
                    means.append(0.0); cis.append(0.0)
                else:
                    m = float(np.mean(arr))
                    s = float(np.std(arr, ddof=1)) if arr.size>1 else 0.0
                    ci = 1.96 * s / math.sqrt(arr.size) if arr.size>1 else 0.0
                    means.append(m); cis.append(ci)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(x, means, yerr=cis)
            ax.set_xticks(x); ax.set_xticklabels(labs)
            ax.axhline(1.0, linestyle="--")
            ax.set_title(f"FT / BASE ratio of Jacobian sensitivities — layer {args.layer_index}")
            fig.tight_layout()
            fig.savefig(outdir / f"jacobian_ratio_ft_over_base_layer{args.layer_index}.png", dpi=170)
            plt.close(fig)
    except Exception as e:
        logging.warning("Failed to plot jacobian_ratio_ft_over_base: %s", e)

    # 3) Overlaid histograms of FT−BASE per prompt for MEAN & VAR (and optionally RANDOM)
    try:
        if merged is not None and not merged.empty:
            diffs = {
                "MEAN": merged["jac_MEAN_FT"] - merged["jac_MEAN_BASE"],
                "VAR":  merged["jac_VAR_mean_FT"] - merged["jac_VAR_mean_BASE"],
                # "RANDOM": merged["jac_RAND_mean_FT"] - merged["jac_RAND_mean_BASE"],  # uncomment if desired
            }
            fig, ax = plt.subplots(figsize=(9,5))
            for lab, series in diffs.items():
                vals = series.dropna().values
                if len(vals) == 0: continue
                ax.hist(vals, bins=30, alpha=0.5, label=lab)
            ax.axvline(0.0, linestyle="--")
            ax.set_title(f"Distribution of FT − BASE Jacobian sensitivities — layer {args.layer_index}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / f"jacobian_diff_hist_layer{args.layer_index}.png", dpi=170)
            plt.close(fig)
    except Exception as e:
        logging.warning("Failed to plot jacobian_diff_hist: %s", e)

    # 4) Scatter plot: BASE vs FT (VAR)
    try:
        if merged is not None and not merged.empty:
            x = merged["jac_VAR_mean_BASE"].values
            y = merged["jac_VAR_mean_FT"].values
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]; y = y[mask]
            fig, ax = plt.subplots(figsize=(6,6))
            ax.scatter(x, y, s=12, alpha=0.6)
            mn = min(x.min(), y.min()); mx = max(x.max(), y.max())
            ax.plot([mn, mx], [mn, mx], linestyle="--")
            ax.set_xlabel("BASE VAR Jacobian")
            ax.set_ylabel(f"FT_{args.tag} VAR Jacobian")
            ax.set_title(f"BASE vs FT (VAR) — layer {args.layer_index}")
            fig.tight_layout()
            fig.savefig(outdir / f"jacobian_scatter_VAR_layer{args.layer_index}.png", dpi=170)
            plt.close(fig)
    except Exception as e:
        logging.warning("Failed to plot jacobian_scatter_VAR: %s", e)

    # 5) Boxplots: VAR sensitivity distributions (BASE vs FT) in one figure
    try:
        if merged is not None and not merged.empty:
            data = [
                merged["jac_VAR_mean_BASE"].dropna().values,
                merged["jac_VAR_mean_FT"].dropna().values
            ]
            fig, ax = plt.subplots(figsize=(7,5))
            ax.boxplot(data, labels=["BASE","FT"])
            ax.set_title(f"VAR Jacobian distribution (boxplot) — layer {args.layer_index}")
            fig.tight_layout()
            fig.savefig(outdir / f"jacobian_var_box_layer{args.layer_index}.png", dpi=170)
            plt.close(fig)
    except Exception as e:
        logging.warning("Failed to plot jacobian_var_box: %s", e)

    # 6) Combined MLP ||h|| bars in one figure (BASE & FT)
    try:
        tags = list(mlp_norm_rows.keys())
        means = []; cis = []; labels=[]
        for tag in tags:
            df = pd.DataFrame(mlp_norm_rows[tag])
            vals = df["mlp_out_norm_mean"].dropna().values
            m = float(np.mean(vals)) if len(vals)>0 else 0.0
            s = float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0
            ci = 1.96 * s / math.sqrt(len(vals)) if len(vals)>1 else 0.0
            means.append(m); cis.append(ci); labels.append(tag)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(labels, means, yerr=cis)
        ax.set_title(f"||MLP(h)||₂ (mean over paraphrases) — BASE vs FT — layer {args.layer_index}")
        fig.tight_layout()
        fig.savefig(outdir / f"mlp_out_norm_combined_bars_layer{args.layer_index}.png", dpi=170)
        plt.close(fig)
    except Exception as e:
        logging.warning("Failed to plot mlp_out_norm_combined_bars: %s", e)

    # 7) PCA overlays (if applicable): EVR and cumulative EVR for BASE vs FT
    if args.jacobian_mode == "pca":
        try:
            base_pca = pd.DataFrame(pca_rows["BASE"])
            ft_pca   = pd.DataFrame(pca_rows[f"FT_{args.tag}"])
            if not base_pca.empty and not ft_pca.empty:
                maxk = args.topk_pca
                evr_cols = [f"evr_{i+1}" for i in range(maxk)]
                base_means = []; ft_means = []
                for c in evr_cols:
                    base_means.append(float(base_pca[c].dropna().mean() if c in base_pca else 0.0))
                    ft_means.append(float(ft_pca[c].dropna().mean() if c in ft_pca else 0.0))
                # Overlay EVR
                fig, ax = plt.subplots(figsize=(8,4))
                xs = np.arange(1, len(base_means)+1)
                ax.plot(xs, base_means, marker="o", label="BASE")
                ax.plot(xs, ft_means, marker="o", label=f"FT_{args.tag}")
                ax.set_xticks(xs); ax.set_xlabel("PC index")
                ax.set_title(f"EVR overlay (mean across prompts) — layer {args.layer_index}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(outdir / f"pca_scree_overlay_layer{args.layer_index}.png", dpi=170)
                plt.close(fig)
                # Cumulative EVR
                base_cum = np.cumsum(base_means)
                ft_cum   = np.cumsum(ft_means)
                fig, ax = plt.subplots(figsize=(8,4))
                ax.plot(xs, base_cum, marker="o", label="BASE")
                ax.plot(xs, ft_cum, marker="o", label=f"FT_{args.tag}")
                ax.set_xticks(xs); ax.set_ylim(0, 1.05)
                ax.set_title(f"Cumulative EVR overlay — layer {args.layer_index}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(outdir / f"pca_cum_evr_overlay_layer{args.layer_index}.png", dpi=170)
                plt.close(fig)
        except Exception as e:
            logging.warning("Failed to plot PCA overlays: %s", e)

    # README (kept + augmented list)
    lines = []
    lines.append("# Paraphrase subspace + Jacobian — outputs")
    lines.append("")
    lines.append("Per-model CSVs (per prompt):")
    lines.append(f"- `BASE_paraphrase_subspace_jacobian_layer{args.layer_index}.csv`")
    lines.append(f"- `FT_{args.tag}_paraphrase_subspace_jacobian_layer{args.layer_index}.csv`")
    lines.append(f"- `BASE_mlp_out_norm_stats_layer{args.layer_index}.csv`")
    lines.append(f"- `FT_{args.tag}_mlp_out_norm_stats_layer{args.layer_index}.csv`")
    if args.jacobian_mode == "pca":
        lines.append(f"- `BASE_pca_stats_layer{args.layer_index}.csv`")
        lines.append(f"- `FT_{args.tag}_pca_stats_layer{args.layer_index}.csv`")
    lines.append("")
    lines.append("Figures (original):")
    lines.append(f"- `jacobian_var_bars_layer{args.layer_index}.png`")
    lines.append(f"- `jacobian_rand_bars_layer{args.layer_index}.png`")
    lines.append(f"- `jacobian_mean_dir_bars_layer{args.layer_index}.png`")
    lines.append(f"- `mlp_out_norm_mean_bars_layer{args.layer_index}.png`")
    if args.jacobian_mode == "pca":
        lines.append(f"- `pca_scree_mean_layer{args.layer_index}.png`")
    lines.append("")
    lines.append("Figures (new, comparative):")
    lines.append(f"- `jacobian_combined_layer{args.layer_index}.png`  (MEAN/VAR/RAND, BASE & FT, colored)")
    lines.append(f"- `jacobian_ratio_ft_over_base_layer{args.layer_index}.png`  (FT/BASE ratio with 95% CI)")
    lines.append(f"- `jacobian_diff_hist_layer{args.layer_index}.png`  (FT−BASE distribution overlays)")
    lines.append(f"- `jacobian_scatter_VAR_layer{args.layer_index}.png`  (per-prompt VAR BASE vs FT, y=x)")
    lines.append(f"- `jacobian_var_box_layer{args.layer_index}.png`  (VAR boxplots BASE vs FT)")
    lines.append(f"- `mlp_out_norm_combined_bars_layer{args.layer_index}.png`  (BASE & FT together)")
    if args.jacobian_mode == "pca":
        lines.append(f"- `pca_scree_overlay_layer{args.layer_index}.png`  (EVR overlay BASE vs FT)")
        lines.append(f"- `pca_cum_evr_overlay_layer{args.layer_index}.png`  (cumulative EVR overlay)")
    lines.append("")
    
    lines.append("Nonlinearity experiment outputs:")
    lines.append(f"- `nl_log_compression_bars_layer{args.layer_index}.png`")
    lines.append(f"- `nl_compression_ratio_hist_overlay_layer{args.layer_index}.png`")
    lines.append(f"- `nl_gate_frac_neg_bars_layer{args.layer_index}.png`, `nl_spars_post_abs_bars_layer{args.layer_index}.png`, `nl_spars_post_adapt_bars_layer{args.layer_index}.png`")
    lines.append(f"- `nl_dir_compression_violin_layer{args.layer_index}.png`")
    lines.append(f"- `nl_log_compression_scatter_layer{args.layer_index}.png`")
    lines.append(f"- `{ 'BASE' }_nl_stats_layer{args.layer_index}.csv`, `FT_{args.tag}_nl_stats_layer{args.layer_index}.csv`")
    lines.append("Interpretation sketch:")
    lines.append("- If `FT` shows **lower** Jacobian on VAR than BASE but similar on MEAN, it supports 'denoising paraphrase variance'.")
    lines.append("- If `FT/BASE < 1` for VAR and ~1 for MEAN, same story. Scatter below y=x for VAR reinforces it.")
    lines.append("- Shifts in `||MLP(h)||` indicate selective amplification/suppression (semantic core vs variance).")
    (outdir / "README.md").write_text("\\n".join(lines), encoding="utf-8")

    logging.getLogger("jacobian").info("Done. CSVs and figures written to %s", outdir)

if __name__ == "__main__":
    main()
