"""
python3 f_finetune/src/compare_script03.py \
  --base_model_path f_finetune/model \
  --ft_model_path f_finetune/outputs_great_nolap/lpr9x_a1_notarg_50k_ft/final \
  --prompts_json_path a_data/alpaca/50k_phrxed.json \
  --output_dir f_finetune/outputs_great_nolap/lpr9x_a1_notarg_50k_ft/mass02/masstest50 \
  --scores_base_json_path c_assess_inf/output/alpaca_answer_scores/gemma-2-2b-it.json \
  --scores_ft_json_path f_finetune/outputs_great_nolap/lpr9x_a1_notarg_50k_ft/scores.json \
  --run_mode aggregate \
  --pairs_per_batch 16 \
  --dtype bfloat16 \
  --limit 50 \
  --focus_on_answer_tokens \
  --compute_pii --pii_focus_on_answer \
  --flash_attention auto \
  --capture_component_activations --capture_attn_qkv --capture_mlp_up_gate \
  --list_params_containing layers.0.mlp.down_proj \
  --inspect_param model.layers.0.mlp.down_proj.weight
"""
import argparse
import json
import logging
import os
import sys
import math
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

# Global toggle used by cosine helpers to avoid referencing 'args' directly
UPCAST_FP32 = False


# Logging
class SafeFormatter(logging.Formatter):
    def format(self, record):
        rec = logging.makeLogRecord(record.__dict__)

        # Coerce non-string messages from warnings/exceptions
        if not isinstance(rec.msg, str):
            try:
                rec.msg = str(rec.msg)
            except Exception:
                rec.msg = "<non-string log message>"

        # If msg has no %-placeholders but args were injected (common with warnings),
        # drop args so logging doesn't attempt old-style formatting.
        if rec.args and isinstance(rec.msg, str) and ('%' not in rec.msg):
            rec.args = ()

        try:
            return super().format(rec)
        except Exception:
            # Last-resort fallback without the scary "MALFORMED LOG" prefix
            return f"[{rec.levelname}] {rec.msg}"

safe_handler = logging.StreamHandler(sys.stdout)
safe_handler.setFormatter(SafeFormatter("%(asctime)s [%(levelname)s] - %(message)s"))
script_logger = logging.getLogger(__name__)
script_logger.setLevel(logging.INFO)
script_logger.handlers.clear()
script_logger.addHandler(safe_handler)
script_logger.propagate = False

transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
transformers_logger.handlers.clear()
transformers_logger.addHandler(safe_handler)
transformers_logger.propagate = False

# Perf knobs
def enable_perf_features(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")  # "medium"/"high" ok on A100
        except Exception:
            pass
        torch.backends.cudnn.allow_tf32 = True
    # FlashAttention / SDPA selection (best-effort; depends on HF + installed libs)
    attn_impl = None
    if args.flash_attention == "auto":
        # Try FA2, else SDPA, else None (HF default)
        for impl in ("flash_attention_2", "sdpa"):
            try:
                attn_impl = impl
                break
            except Exception:
                continue
    elif args.flash_attention in ("flash_attention_2", "sdpa"):
        attn_impl = args.flash_attention
    return attn_impl

# Helpers
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: [seq, hidden] or [batch, seq, hidden]
    mask: [seq] or [batch, seq] with 1/True for valid tokens
    returns: [hidden] or [batch, hidden]
    """
    if x.dim() == 2:
        mask = mask.to(x.device).float()
        denom = mask.sum().clamp_min(1.0)
        return (x * mask.unsqueeze(-1)).sum(dim=0) / denom
    elif x.dim() == 3:
        mask = mask.to(x.device).float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom
    else:
        raise ValueError("masked_mean expects 2D or 3D tensor.")

def robust_mad_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return 0.6745 * (x - med) / mad

def cosine(a: torch.Tensor, b: torch.Tensor, upcast_fp32: bool = False) -> float:
    """
    Cosine similarity with optional fp32 upcast.
    Default keeps original dtype (e.g., bf16) per user preference.
    """
    if upcast_fp32:
        a = a.float()
        b = b.float()
    a = a.view(1, -1)
    b = b.view(1, -1)
    an = torch.linalg.vector_norm(a, dim=1)
    bn = torch.linalg.vector_norm(b, dim=1)
    denom = (an * bn).clamp_min(1e-12)
    val = torch.sum(a * b, dim=1).div(denom)
    return torch.clamp(val, -1.0, 1.0).item()

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def build_user_message(instruction: str, inp: str) -> str:
    return instruction if not inp else f"{instruction}\n\nInput:\n{inp}"

# Simple answer cache (RAM-only)
class AnswerCache:
    def __init__(self):
        self._cache = {}
    def key(self, model_id: str, instruction: str, inp: str, max_new_tokens: int, temperature: float, top_p: float):
        return (model_id, instruction, inp, max_new_tokens, temperature, top_p)
    def get(self, *k):
        return self._cache.get(k, None)
    def put(self, *k, value: str):
        self._cache[k] = value

@torch.no_grad()
def generate_answer_text(model: PreTrainedModel, tokenizer: AutoTokenizer, instruction: str, inp: str,
                         device: str, max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0,
                         cache: Optional[AnswerCache] = None, model_id: str = "") -> str:
    if cache is not None:
        k = cache.key(model_id, instruction, inp, max_new_tokens, temperature, top_p)
        v = cache.get(*k)
        if v is not None:
            return v
    messages = [{"role": "user", "content": build_user_message(instruction, inp)}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
    do_sample = (temperature and float(temperature) > 0.0) or (top_p and float(top_p) < 1.0)
    gen_kwargs = dict(max_new_tokens=max_new_tokens,
                       do_sample=bool(do_sample),
                       eos_token_id=tokenizer.eos_token_id)
    if do_sample:
        gen_kwargs.update(dict(temperature=temperature, top_p=top_p))
    #out = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), **gen_kwargs)
    out = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                     use_cache=True, **gen_kwargs)
    gen_ids = out[0, input_ids.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if cache is not None:
        cache.put(*k, value=text)
    return text

def tokenise_with_segments(tokenizer: AutoTokenizer, instruction: str, inp: str, device: str, answer_text: Optional[str] = None):
    """
    Returns dict with:
      - input_ids, attention_mask
      - boolean masks for segments (shape [1, seq] torch.bool on device):
          seg_instruction, seg_input, seg_asst_prefix, seg_answer
        * seg_answer marks ONLY produced answer tokens (if answer_text is provided).
    Uses chat template if available, else falls back to explicit sections.
    """
    user_msg = build_user_message(instruction, inp)
    try:
        messages = [{"role": "user", "content": user_msg}]
        plain_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")
        asst_ids  = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

        seg_len_plain = plain_ids.shape[1]
        seg_len_asst  = asst_ids.shape[1]
        asst_prefix_len = seg_len_asst - seg_len_plain

        if answer_text:
            ans_ids = tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
            input_ids = torch.cat([asst_ids, ans_ids], dim=1).to(device)
        else:
            input_ids = asst_ids.to(device)

        attention_mask = torch.ones_like(input_ids)

        instr_only = tokenizer.apply_chat_template([{"role":"user","content": instruction}], tokenize=True, add_generation_prompt=False, return_tensors="pt")
        len_instr_only = instr_only.shape[1]
        if inp:
            user_w_input = tokenizer.apply_chat_template([{"role":"user","content": user_msg}], tokenize=True, add_generation_prompt=False, return_tensors="pt")
            len_user_w_input = user_w_input.shape[1]
            input_len = max(0, len_user_w_input - len_instr_only)
        else:
            input_len = 0

        total_len = input_ids.shape[1]
        seg_instruction = torch.zeros(total_len, dtype=torch.bool)
        seg_input       = torch.zeros(total_len, dtype=torch.bool)
        seg_asst_prefix = torch.zeros(total_len, dtype=torch.bool)
        seg_answer      = torch.zeros(total_len, dtype=torch.bool)

        seg_instruction[:min(len_instr_only, total_len)] = True
        if input_len > 0:
            start = len_instr_only
            end = min(start + input_len, total_len)
            seg_input[start:end] = True

        if asst_prefix_len > 0:
            asst_start = seg_len_plain
            asst_end   = min(seg_len_asst, total_len)
            seg_asst_prefix[asst_start:asst_end] = True

        if answer_text:
            ans_start = seg_len_asst
            ans_end   = total_len
            if ans_start < ans_end:
                seg_answer[ans_start:ans_end] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask.to(device),
            "seg_instruction": seg_instruction.unsqueeze(0).to(device),
            "seg_input": seg_input.unsqueeze(0).to(device),
            "seg_asst_prefix": seg_asst_prefix.unsqueeze(0).to(device),
            "seg_answer": seg_answer.unsqueeze(0).to(device),
        }

    except Exception:
        # Fallback
        prefix_instr = "### Instruction:\n"
        prefix_input = "### Input:\n"
        prefix_ans   = "### Answer:\n"

        text = prefix_instr + instruction + "\n\n"
        instr_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]

        if inp:
            text_input = prefix_input + inp + "\n\n"
            input_ids_extra = tokenizer(text_input, add_special_tokens=False, return_tensors="pt")["input_ids"]
            prompt_ids = torch.cat([instr_ids, input_ids_extra], dim=1)
            instr_len = instr_ids.shape[1]
            input_len = input_ids_extra.shape[1]
        else:
            prompt_ids = instr_ids
            instr_len = prompt_ids.shape[1]
            input_len = 0

        if answer_text:
            text_ans = prefix_ans + answer_text
            ans_ids = tokenizer(text_ans, add_special_tokens=False, return_tensors="pt")["input_ids"]
            input_ids = torch.cat([prompt_ids, ans_ids], dim=1)
        else:
            input_ids = prompt_ids

        total_len = input_ids.shape[1]
        seg_instruction = torch.zeros(total_len, dtype=torch.bool)
        seg_input       = torch.zeros(total_len, dtype=torch.bool)
        seg_asst_prefix = torch.zeros(total_len, dtype=torch.bool)
        seg_answer      = torch.zeros(total_len, dtype=torch.bool)

        seg_instruction[:instr_len] = True
        if input_len > 0:
            seg_input[instr_len:instr_len+input_len] = True

        if answer_text:
            ans_start = prompt_ids.shape[1]
            seg_answer[ans_start:total_len] = True

        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "seg_instruction": seg_instruction.unsqueeze(0).to(device),
            "seg_input": seg_input.unsqueeze(0).to(device),
            "seg_asst_prefix": seg_asst_prefix.unsqueeze(0).to(device),
            "seg_answer": seg_answer.unsqueeze(0).to(device),
        }

# Component hooks (optional; heavier)
class ComponentProbe:
    """
    Captures per-layer outputs of specific submodules:
      - self_attn.{q_proj,k_proj,v_proj,o_proj} (if enabled)
      - mlp.{up_proj,gate_proj,down_proj}      (if enabled)
    """
    def __init__(self, model: PreTrainedModel,
                 capture_attn_o: bool = True,
                 capture_mlp_down: bool = True,
                 capture_attn_qkv: bool = False,
                 capture_mlp_up_gate: bool = False):
        self.model = model
        self.capture_attn_o = capture_attn_o
        self.capture_mlp_down = capture_mlp_down
        self.capture_attn_qkv = capture_attn_qkv
        self.capture_mlp_up_gate = capture_mlp_up_gate
        self._handles = []
        self.attn_o: Dict[int, torch.Tensor] = {}
        self.attn_q: Dict[int, torch.Tensor] = {}
        self.attn_k: Dict[int, torch.Tensor] = {}
        self.attn_v: Dict[int, torch.Tensor] = {}
        self.mlp_down: Dict[int, torch.Tensor] = {}
        self.mlp_up: Dict[int, torch.Tensor] = {}
        self.mlp_gate: Dict[int, torch.Tensor] = {}

    def _make_hook(self, store_dict: Dict[int, torch.Tensor], layer_idx: int):
        def hook(module, inp, out):
            # out shape: [batch, seq, hidden] (expected)
            try:
                store_dict[layer_idx] = out.detach()
            except Exception:
                pass
        return hook

    def __enter__(self):
        self.attn_o.clear(); self.mlp_down.clear()
        # Walk layers
        if not hasattr(self.model, 'model') or not hasattr(self.model.model, 'layers'):
            raise TypeError("Model does not have the expected 'model.layers' structure.")
        for i, layer in enumerate(self.model.model.layers):
            # Try to find submodules robustly by name scanning
            if self.capture_attn_o:
                for name, mod in layer.named_modules():
                    # common names: self_attn.o_proj, attention.o_proj
                    if name.endswith("o_proj"):
                        h = mod.register_forward_hook(self._make_hook(self.attn_o, i))
                        self._handles.append(h)
                        break
            if self.capture_mlp_down:
                for name, mod in layer.named_modules():
                    if name.endswith("down_proj"):
                        h = mod.register_forward_hook(self._make_hook(self.mlp_down, i))
                        self._handles.append(h)
                        break

            if getattr(self, "capture_attn_qkv", False):
                for name, mod in layer.named_modules():
                    if name.endswith("q_proj"):
                        self._handles.append(mod.register_forward_hook(self._make_hook(self.attn_q, i)))
                    elif name.endswith("k_proj"):
                        self._handles.append(mod.register_forward_hook(self._make_hook(self.attn_k, i)))
                    elif name.endswith("v_proj"):
                        self._handles.append(mod.register_forward_hook(self._make_hook(self.attn_v, i)))

            if getattr(self, "capture_mlp_up_gate", False):
                for name, mod in layer.named_modules():
                    if name.endswith("up_proj"):
                        self._handles.append(mod.register_forward_hook(self._make_hook(self.mlp_up, i)))
                    elif name.endswith("gate_proj"):
                        self._handles.append(mod.register_forward_hook(self._make_hook(self.mlp_gate, i)))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            try: h.remove()
            except Exception: pass
        self._handles = []

# Activation capture (per-layer hidden states)
class ActivationExtractor:
    def __init__(self, model: PreTrainedModel):
        self._model = model
        self.hidden_states = {}
        self._hook_handles = []
    def _hook_fn(self, layer_idx: int):
        def hook(module, input, output):
            # output[0] expected: hidden states [batch, seq, hidden]
            self.hidden_states[layer_idx] = output[0].detach()
        return hook
    def __enter__(self):
        self.hidden_states.clear()
        if not hasattr(self._model, 'model') or not hasattr(self._model.model, 'layers'):
            raise TypeError("Model does not have the expected 'model.layers' structure.")
        for i, layer in enumerate(self._model.model.layers):
            self._hook_handles.append(layer.register_forward_hook(self._hook_fn(i)))
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self._hook_handles:
            try: handle.remove()
            except Exception: pass
        self._hook_handles.clear()

# Batching utilities
@dataclass
class EncodedExample:
    input_ids: torch.Tensor         # [1, seq]
    attention_mask: torch.Tensor    # [1, seq]
    seg_instruction: torch.Tensor   # [1, seq] bool
    seg_input: torch.Tensor         # [1, seq] bool
    seg_asst_prefix: torch.Tensor   # [1, seq] bool
    seg_answer: torch.Tensor        # [1, seq] bool

def collate_batch(examples: List[EncodedExample], device: str) -> EncodedExample:
    # Pad to max length
    max_len = max(ex.input_ids.shape[1] for ex in examples)
    def pad(t, pad_val=0):
        pad_width = max_len - t.shape[1]
        if pad_width == 0:
            return t
        if t.dtype == torch.bool:
            pad_fill = torch.zeros((t.shape[0], pad_width), dtype=torch.bool, device=t.device)
        else:
            pad_fill = torch.full((t.shape[0], pad_width), pad_val, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad_fill], dim=1)
    input_ids = torch.cat([pad(ex.input_ids) for ex in examples], dim=0).to(device)
    attention_mask = torch.cat([pad(ex.attention_mask, 1) for ex in examples], dim=0).to(device)
    seg_instruction = torch.cat([pad(ex.seg_instruction) for ex in examples], dim=0).to(device)
    seg_input       = torch.cat([pad(ex.seg_input) for ex in examples], dim=0).to(device)
    seg_asst_prefix = torch.cat([pad(ex.seg_asst_prefix) for ex in examples], dim=0).to(device)
    seg_answer      = torch.cat([pad(ex.seg_answer) for ex in examples], dim=0).to(device)
    return EncodedExample(input_ids, attention_mask, seg_instruction, seg_input, seg_asst_prefix, seg_answer)

def encode_pair(tokenizer: AutoTokenizer, instruction: str, paraphrase: str, inp: str, device: str,
                ans_o: Optional[str], ans_p: Optional[str]) -> Tuple[EncodedExample, EncodedExample]:
    enc_o = tokenise_with_segments(tokenizer, instruction, inp, device, answer_text=ans_o)
    enc_p = tokenise_with_segments(tokenizer, paraphrase,  inp, device, answer_text=ans_p)
    ex_o = EncodedExample(**enc_o)
    ex_p = EncodedExample(**enc_p)
    return ex_o, ex_p

# Pairwise metrics (batched)
def cosine_by_layer_masked(hs: Dict[int, torch.Tensor], idx_a: int, idx_b: int,
                           mask_a: torch.Tensor, mask_b: torch.Tensor) -> Dict[int, float]:
    sims = {}
    for l, H in hs.items():
        # H: [B, S, H]
        a = masked_mean(H[idx_a], mask_a[idx_a].bool())
        b = masked_mean(H[idx_b], mask_b[idx_b].bool())
        sims[l] = cosine(a, b, upcast_fp32=UPCAST_FP32)
    return sims

def l2_by_layer_masked(hs: Dict[int, torch.Tensor], idx_a: int, idx_b: int,
                       mask_a: torch.Tensor, mask_b: torch.Tensor) -> Dict[int, float]:
    dists = {}
    for l, H in hs.items():
        a = masked_mean(H[idx_a], mask_a[idx_a].bool()).float()
        b = masked_mean(H[idx_b], mask_b[idx_b].bool()).float()
        dists[l] = torch.linalg.norm(a - b).item()
    return dists

def norms_by_layer_masked(hs: Dict[int, torch.Tensor], idx: int, mask: torch.Tensor) -> Dict[int, float]:
    norms = {}
    for l, H in hs.items():
        v = masked_mean(H[idx], mask[idx].bool())
        norms[l] = torch.linalg.norm(v.float()).item()
    return norms

def component_norms(component_map: Dict[int, torch.Tensor], idx: int, mask: torch.Tensor) -> Dict[int, float]:
    norms = {}
    for l, T in component_map.items():
        v = masked_mean(T[idx], mask[idx].bool())
        norms[l] = torch.linalg.norm(v.float()).item()
    return norms

def component_cosine_by_layer(component_map: Dict[int, torch.Tensor],
                              idx_a: int, idx_b: int,
                              mask_a: torch.Tensor, mask_b: torch.Tensor) -> Dict[int, float]:
    """
    Cosine similarity between masked means of component outputs for items idx_a and idx_b.
    component_map[l] has shape [B, S, H] for each layer l.
    """
    sims = {}
    for l, T in component_map.items():
        va = masked_mean(T[idx_a], mask_a[idx_a].bool())
        vb = masked_mean(T[idx_b], mask_b[idx_b].bool())
        sims[l] = cosine(va, vb, upcast_fp32=UPCAST_FP32)
    return sims

# Weight ΔW analysis (faster, component/top-k, tqdm)
def module_bucket(name: str) -> Tuple[str, str, int]:
    """
    Return (scope, module_bucket, layer_index)
    """
    layer_idx = -1
    if 'layers.' in name:
        try:
            layer_idx = int(name.split('layers.')[1].split('.')[0])
        except Exception:
            layer_idx = -1

    if 'q_proj' in name: return ('layer', 'attn_q_proj', layer_idx)
    if 'k_proj' in name: return ('layer', 'attn_k_proj', layer_idx)
    if 'v_proj' in name: return ('layer', 'attn_v_proj', layer_idx)
    if 'o_proj' in name: return ('layer', 'attn_o_proj', layer_idx)

    if 'up_proj'   in name: return ('layer', 'mlp_up_proj',   layer_idx)
    if 'gate_proj' in name: return ('layer', 'mlp_gate_proj', layer_idx)
    if 'down_proj' in name: return ('layer', 'mlp_down_proj', layer_idx)

    if 'norm' in name: return ('layer', 'norms', layer_idx)
    if ('embed' in name) or ('tok_embeddings' in name) or ('embed_tokens' in name) or ('wte' in name):
        return ('global', 'embeddings', -1)
    if 'lm_head' in name: return ('global', 'lm_head', -1)
    return ('other', 'other', layer_idx)

@torch.no_grad()
def analyze_and_plot_weight_deltas(base_model: PreTrainedModel, ft_model: PreTrainedModel, output_path: Path,
                                   compute_svd: bool = False, topk: int = 5, fast_mode: bool = True,
                                   include_buckets: Optional[List[str]] = None, args=None):
    script_logger.info("Comparing model weights layer by layer (detailed)...")
    base_params = dict(base_model.named_parameters())
    ft_params = dict(ft_model.named_parameters())

    rows = []
    names = [n for n in ft_params.keys() if n in base_params]
    # Optional filtering for speed
    if include_buckets:
        filtered = []
        for n in names:
            _, b, _ = module_bucket(n)
            if b in include_buckets:
                filtered.append(n)
        if filtered:
            names = filtered

    with tqdm(total=len(names), desc="ΔW params", leave=False) as pbar:
        for name in names:
            ft_param = ft_params[name]
            b = base_params[name]
            if ft_param.shape != b.shape:
                pbar.update(1)
                continue
            # Compute diff on device, cast minimally when needed
            diff = (ft_param - b)

            # for numeric stability cast to float32 for norm, but avoid host transfer
            delta_frob = torch.linalg.norm(diff.float()).item()
            n_params = diff.numel()
            mean_abs = diff.abs().mean().item()

            base_fro = torch.linalg.norm(b.float()).item()
            ft_fro   = torch.linalg.norm(ft_param.float()).item()

            # Cosine metrics between weights/tensors (keep dtype unless flag says otherwise)
            b_flat  = b.detach()
            ft_flat = ft_param.detach()
            d_flat  = diff.detach()

            #if args.cosine_upcast_fp32:
            #    b_flat  = b_flat.float()
            #    ft_flat = ft_flat.float()
            #    d_flat  = d_flat.float()

            b_flat  = b_flat.view(1, -1)
            ft_flat = ft_flat.view(1, -1)
            d_flat  = d_flat.view(1, -1)

            def _safe_cos(u, v):
                # ALWAYS compute in fp32 and clamp
                u = u.float()
                v = v.float()
                un = torch.linalg.vector_norm(u, dim=1)
                vn = torch.linalg.vector_norm(v, dim=1)
                denom = (un * vn).clamp_min(1e-12)
                val = torch.sum(u * v, dim=1).div(denom)
                return float(torch.clamp(val, -1.0, 1.0).item())

            cos_base_ft    = _safe_cos(b_flat,  ft_flat)
            cos_delta_base = _safe_cos(d_flat,  b_flat)
            cos_delta_ft   = _safe_cos(d_flat,  ft_flat)

            scope, bucket, layer_idx = module_bucket(name)
            svd_vals = []
            energy_ratio_topk = None

            if compute_svd and diff.dim() == 2 and min(diff.shape) >= topk and max(diff.shape) <= 16384:
                try:
                    sv = torch.linalg.svdvals(diff.float()).detach().cpu().numpy()
                    sv_sorted = np.sort(sv)[::-1]
                    top = sv_sorted[:topk]
                    svd_vals = top.tolist()
                    energy_ratio_topk = float((top**2).sum() / ((sv_sorted**2).sum() + 1e-9))
                except Exception as e:
                    script_logger.warning(f"SVD failed for {name} (shape {list(diff.shape)}): {e}")

            rows.append({
                "param_name": name,
                "scope": scope,
                "bucket": bucket,
                "layer_idx": layer_idx,
                "shape": list(ft_param.shape),
                "n_params": n_params,
                "delta_fro": delta_frob,
                "mean_abs": mean_abs,
                "delta_per_param": (delta_frob / max(1, n_params)),
                # dimensionally-consistent RMS change per weight
                "delta_rmse": (delta_frob / max(1, math.sqrt(n_params))),
                # cosine fields
                "cos_base_ft": cos_base_ft,
                "cos_delta_base": cos_delta_base,
                "cos_delta_ft": cos_delta_ft,
                "svd_topk": svd_vals,
                "topk_energy_ratio": energy_ratio_topk,
                "base_fro": base_fro,
                "ft_fro": ft_fro,
            })
            pbar.update(1)

    df = pd.DataFrame(rows)
    if df.empty:
        script_logger.warning("No comparable parameters for ΔW analysis.")
        return
    
    # include embeddings in per-layer views and add an 'E' tick ---
    # Treat embeddings as a pseudo-layer -1 (already set in module_bucket). Build a convenience mask.
    is_layer_or_embedding = (df["layer_idx"] >= 0) | (df["bucket"] == "embeddings")

    def _format_layer_axis(ax):
        """Make sure -1 is shown as 'E' on x-axis."""
        xs = list(sorted(set(int(x) for x in df.loc[is_layer_or_embedding, "layer_idx"].unique())))
        if -1 in xs:
            # keep -1 first
            xs = [-1] + [x for x in xs if x != -1]
        ax.set_xticks(xs)
        ax.set_xticklabels([("E" if x == -1 else str(x)) for x in xs])
    
    # Per-layer absolute ||W||_F (sum over params in that layer)
    agg_layer_abs = (
        df[is_layer_or_embedding]
        .groupby("layer_idx")[["base_fro","ft_fro"]]
        .sum()
        .reset_index()
    )

    plt.figure()
    plt.plot(agg_layer_abs["layer_idx"], agg_layer_abs["base_fro"], label="Base ||W||₍F₎")
    plt.plot(agg_layer_abs["layer_idx"], agg_layer_abs["ft_fro"],   label="FT ||W||₍F₎")
    plt.xlabel("Layer"); plt.ylabel("Frobenius Norm (sum)"); plt.title("Absolute Weight Size per Layer")
    plt.legend(); _format_layer_axis(plt.gca())
    plt.tight_layout()
    plt.savefig(output_path / "weights_abs_fro_per_layer.png", dpi=180)
    plt.close()

    # By component bucket (q/k/v/o, mlp up/gate/down, etc.)
    agg_bucket_abs = (
        df.groupby("bucket")[["base_fro","ft_fro"]]
        .sum()
        .sort_values("ft_fro", ascending=False)
        .reset_index()
    )

    plt.figure(figsize=(12,6))
    y = np.arange(len(agg_bucket_abs))
    plt.barh(y-0.2, agg_bucket_abs["base_fro"], height=0.4, label="Base")
    plt.barh(y+0.2, agg_bucket_abs["ft_fro"],   height=0.4, label="FT")
    plt.yticks(y, agg_bucket_abs["bucket"])
    plt.xlabel("Σ ||W||₍F₎"); plt.title("Absolute Weight Size by Module Bucket")
    plt.legend(); plt.tight_layout()
    plt.savefig(output_path / "weights_abs_fro_by_bucket.png", dpi=180)
    plt.close()

    df.to_csv(output_path / "weight_deltas_detailed.csv", index=False)

    # parameter-count weighted cosine similarity per (layer × component)
    # We weight cosines by n_params so giant matrices count proportionally.
    sub_df = df[is_layer_or_embedding].copy()
    if not sub_df.empty:
        def _wavg(g, key):
            vals = g[key].to_numpy()
            wts  = g["n_params"].to_numpy()
            return float(np.average(vals, weights=wts)) if wts.sum() > 0 else float(np.nan)

        # Weighted averages for the three cosine metrics you already compute
        wtd = (
            sub_df
            .groupby(["layer_idx", "bucket"])
            .apply(lambda g: pd.Series({
                "wtd_cos_base_ft":    _wavg(g, "cos_base_ft"),
                "wtd_cos_delta_base": _wavg(g, "cos_delta_base"),
                "wtd_cos_delta_ft":   _wavg(g, "cos_delta_ft"),
                "n_params_sum":       int(g["n_params"].sum()),
            }))
            .reset_index()
            .sort_values(["bucket", "layer_idx"])
        )
        # Distances (1 − cosine) so easier to read
        # Clamp before taking (1 - cos)
        for col in ("wtd_cos_base_ft", "wtd_cos_delta_base", "wtd_cos_delta_ft"):
            wtd[col] = wtd[col].clip(-1.0, 1.0)

        wtd["one_minus_cos_base_ft"]    = 1.0 - wtd["wtd_cos_base_ft"]
        wtd["one_minus_cos_delta_base"] = 1.0 - wtd["wtd_cos_delta_base"]
        wtd["one_minus_cos_delta_ft"]   = 1.0 - wtd["wtd_cos_delta_ft"]

        # Save CSVs
        wtd.to_csv(output_path / "weights_cosine_per_layer_component__weighted.csv", index=False)

        # Heatmap of (1 − cosine) for base vs ft alignment
        heat = wtd.pivot(index="bucket", columns="layer_idx", values="one_minus_cos_base_ft").fillna(0.0)
        if not heat.empty:
            cols = sorted(heat.columns)
            if -1 in cols:
                cols = [-1] + [c for c in cols if c != -1]
            heat = heat[cols]

        if not heat.empty:
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.heatmap(heat, cmap="viridis", cbar_kws={'label': '1 − cos(W_base, W_ft)'})
            ax.set_title("Weighted (1 − cosine) by Component × Layer")
            ax.set_xlabel("Layer"); ax.set_ylabel("Component")
            plt.tight_layout(); fig.savefig(output_path / "weights_one_minus_cosine_heatmap.png", dpi=180); plt.close(fig)

        # Line plots: (1 − cosine) per component across layers
        if not getattr(args, "skip_weights_one_minus_cosine_per_layer_plots", False):
            for bname in sorted(wtd["bucket"].unique()):
                sub = wtd[wtd["bucket"] == bname].sort_values("layer_idx")
                plt.figure()
                plt.plot(sub["layer_idx"], sub["one_minus_cos_base_ft"],    label="1 − cos(W_base, W_ft)")
                plt.plot(sub["layer_idx"], sub["one_minus_cos_delta_base"], label="1 − cos(ΔW, W_base)")
                plt.plot(sub["layer_idx"], sub["one_minus_cos_delta_ft"],   label="1 − cos(ΔW, W_ft)")
                plt.xlabel("Layer"); plt.ylabel("1 − cosine")
                plt.title(f"Directional Change (weighted) — {bname}")
                plt.legend(); _format_layer_axis(plt.gca()); plt.tight_layout()
                plt.savefig(output_path / f"weights_one_minus_cosine_per_layer__{bname}.png", dpi=180)
                plt.close()

        # overall weighted cosine by layer (collapsed across components)
        overall = (
            wtd.groupby("layer_idx")
            .apply(lambda g: pd.Series({
                "wtd_cos_base_ft":    np.average(g["wtd_cos_base_ft"],    weights=g["n_params_sum"]),
                "wtd_cos_delta_base": np.average(g["wtd_cos_delta_base"], weights=g["n_params_sum"]),
                "wtd_cos_delta_ft":   np.average(g["wtd_cos_delta_ft"],   weights=g["n_params_sum"]),
            }))
            .reset_index()
            .sort_values("layer_idx")
        )
        overall["one_minus_cos_base_ft"]    = 1.0 - overall["wtd_cos_base_ft"]
        overall["one_minus_cos_delta_base"] = 1.0 - overall["wtd_cos_delta_base"]
        overall["one_minus_cos_delta_ft"]   = 1.0 - overall["wtd_cos_delta_ft"]
        overall.to_csv(output_path / "weights_cosine_per_layer__overall_weighted.csv", index=False)

        if not getattr(args, "skip_weights_one_minus_cosine_per_layer_plots", False):
            plt.figure()
            plt.plot(overall["layer_idx"], overall["one_minus_cos_base_ft"],    label="1 − cos(W_base, W_ft)")
            plt.plot(overall["layer_idx"], overall["one_minus_cos_delta_base"], label="1 − cos(ΔW, W_base)")
            plt.plot(overall["layer_idx"], overall["one_minus_cos_delta_ft"],   label="1 − cos(ΔW, W_ft)")
            plt.xlabel("Layer"); plt.ylabel("1 − cosine")
            plt.title("Directional Change (weighted) — Overall")
            plt.legend(); _format_layer_axis(plt.gca()); plt.tight_layout()
            plt.savefig(output_path / "weights_one_minus_cosine_per_layer__overall.png", dpi=180)
            plt.close()

    # per-component, per-layer absolute/Δ/relative metrics
    layer_bucket_abs = (
        df[is_layer_or_embedding]
        .groupby(["layer_idx", "bucket"])[["base_fro", "ft_fro", "delta_fro"]]
        .sum()
        .reset_index()
    )

    # Save long + pivots for convenience
    layer_bucket_abs.to_csv(output_path / "weights_abs_fro_per_layer_component_long.csv", index=False)

    pivot_base_abs  = layer_bucket_abs.pivot(index="layer_idx", columns="bucket", values="base_fro").fillna(0.0)
    pivot_ft_abs    = layer_bucket_abs.pivot(index="layer_idx", columns="bucket", values="ft_fro").fillna(0.0)
    pivot_delta_abs = layer_bucket_abs.pivot(index="layer_idx", columns="bucket", values="delta_fro").fillna(0.0)
    pivot_rel_delta = (pivot_delta_abs / (pivot_base_abs + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pivot_base_abs.to_csv(output_path / "weights_abs_fro_per_layer_component__base.csv")
    pivot_ft_abs.to_csv(output_path   / "weights_abs_fro_per_layer_component__ft.csv")
    pivot_delta_abs.to_csv(output_path/ "weights_delta_fro_per_layer_component.csv")
    pivot_rel_delta.to_csv(output_path/ "weights_rel_delta_per_layer_component.csv")

    # One plot per component: absolute size (Base vs FT) by layer
    for bname in sorted(layer_bucket_abs["bucket"].unique()):
        sub = layer_bucket_abs[layer_bucket_abs["bucket"] == bname].sort_values("layer_idx")

        # Absolute ||W||_F per layer (Base vs FT)
        plt.figure()
        plt.plot(sub["layer_idx"], sub["base_fro"], label="Base ||W||₍F₎")
        plt.plot(sub["layer_idx"], sub["ft_fro"],   label="FT ||W||₍F₎")
        plt.xlabel("Layer"); plt.ylabel("Frobenius Norm")
        plt.title(f"Absolute Weight Size per Layer — {bname}")
        plt.legend()
        _format_layer_axis(plt.gca())
        plt.tight_layout()
        plt.savefig(output_path / f"weights_abs_fro_per_layer__{bname}.png", dpi=180)
        plt.close()

        # absolute difference (FT - Base) to make tiny changes visible
        diff = sub["ft_fro"] - sub["base_fro"]
        plt.figure()
        plt.plot(sub["layer_idx"], diff, label="FT − Base (||W||₍F₎)", linewidth=2)
        plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
        plt.xlabel("Layer"); plt.ylabel("Δ Frobenius Norm")
        plt.title(f"Absolute Weight *Difference* per Layer — {bname}")
        plt.legend()
        _format_layer_axis(plt.gca())
        plt.tight_layout()
        plt.savefig(output_path / f"weights_abs_diff_per_layer__{bname}.png", dpi=180)
        plt.close()

        # Absolute change ||ΔW||_F per layer
        if not getattr(args, "skip_weights_delta_per_layer_plots", False):
            plt.figure()
            plt.plot(sub["layer_idx"], sub["delta_fro"], label="||ΔW||₍F₎ (FT − Base)")
            plt.xlabel("Layer"); plt.ylabel("Frobenius Norm of Change")
            plt.title(f"Weight Change Size per Layer — {bname}")
            plt.legend(); _format_layer_axis(plt.gca()); plt.tight_layout()
            plt.savefig(output_path / f"weights_delta_fro_per_layer__{bname}.png", dpi=180)
            plt.close()

        # Relative change ||ΔW||_F / ||W_base||_F per layer
        if not getattr(args, "skip_weights_rel_delta_per_layer_plots", False):
            rel = (sub["delta_fro"] / np.maximum(sub["base_fro"], 1e-12))
            plt.figure()
            plt.plot(sub["layer_idx"], rel, label="||ΔW||₍F₎ / ||W_base||₍F₎")
            plt.xlabel("Layer"); plt.ylabel("Relative Change")
            plt.title(f"Relative Weight Change per Layer — {bname}")
            plt.legend(); _format_layer_axis(plt.gca()); plt.tight_layout()
            plt.savefig(output_path / f"weights_rel_delta_per_layer__{bname}.png", dpi=180)
            plt.close()

        # percent change (zoomed)
        rel = 100.0 * (sub["ft_fro"] - sub["base_fro"]) / np.maximum(sub["base_fro"], 1e-12)
        plt.figure()
        plt.plot(sub["layer_idx"], rel, label="%Δ ||W||₍F₎ (FT vs Base)", linewidth=2)
        # auto-zoom around the data range with a small margin
        m, M = float(rel.min()), float(rel.max())
        pad = max(0.1, 0.15 * (M - m))
        plt.ylim(m - pad, M + pad)
        plt.axhline(0, color="k", linestyle="--", linewidth=1)
        plt.xlabel("Layer"); plt.ylabel("Percent change (%)")
        plt.title(f"Relative Weight Size Change per Layer — {bname}")
        plt.legend()
        _format_layer_axis(plt.gca())
        plt.tight_layout()
        plt.savefig(output_path / f"weights_abs_percent_change_per_layer__{bname}.png", dpi=180)
        plt.close()

    # small textual summary for quick interpretation
    summary_rows = []
    for bname in sorted(layer_bucket_abs["bucket"].unique()):
        sub = layer_bucket_abs[layer_bucket_abs["bucket"] == bname]
        base = sub["base_fro"].to_numpy()
        dlt  = sub["delta_fro"].to_numpy()
        rel  = 100.0 * dlt / np.maximum(base, 1e-12)
        summary_rows.append({
            "bucket": bname,
            "mean_%Δ_abs_norm": float(np.mean(rel)),
            "max_%Δ_abs_norm":  float(np.max(rel)),
            "mean_ΔW_F":        float(np.mean(dlt)),
            "sum_ΔW_F":         float(np.sum(dlt))
        })
    pd.DataFrame(summary_rows).to_csv(output_path / "weights_change_summary_by_bucket.csv", index=False)

    # OPTIONAL: add a "raw (unnormalized) alignment" metric and plots
    # Cosine is normalized by definition, but an unnormalized analogue is the Frobenius
    # inner product <W_base, W_ft> = ||W_base||_F * ||W_ft||_F * cos_base_ft.
    df["base_ft_alignment"] = df["cos_base_ft"] * df["base_fro"] * df["ft_fro"]
    align_layer_bucket = (
        df[is_layer_or_embedding]
        .groupby(["layer_idx", "bucket"])["base_ft_alignment"]
        .sum()
        .reset_index()
    )
    align_layer_bucket.to_csv(output_path / "weights_alignment_per_layer_component.csv", index=False)

    for bname in sorted(align_layer_bucket["bucket"].unique()):
        sub = align_layer_bucket[align_layer_bucket["bucket"] == bname].sort_values("layer_idx")
        plt.figure()
        plt.plot(sub["layer_idx"], sub["base_ft_alignment"], label="<W_base, W_ft>")
        plt.xlabel("Layer"); plt.ylabel("Frobenius Inner Product")
        plt.title(f"Raw Alignment (not normalized) per Layer — {bname}")
        plt.legend(); _format_layer_axis(plt.gca()); plt.tight_layout()
        plt.savefig(output_path / f"weights_alignment_per_layer__{bname}.png", dpi=180)
        plt.close()


    # per-layer × component
    layer_comp = df[df["layer_idx"] >= 0].groupby(["layer_idx","bucket"])["delta_fro"].sum().reset_index()
    layer_comp_pivot = layer_comp.pivot(index="layer_idx", columns="bucket", values="delta_fro").fillna(0.0)
    layer_comp.to_csv(output_path / "weight_deltas_per_layer_component_long.csv", index=False)
    layer_comp_pivot.to_csv(output_path / "weight_deltas_per_layer_component.csv", index=True)

    # per-layer × component cosine (WEIGHTED by parameter count)
    if not df.empty and "cos_base_ft" in df.columns:
        g = df[df["layer_idx"] >= 0].groupby(["layer_idx","bucket"])

        def _wavg_cos(series, weights):
            # both are pandas Series
            v = np.asarray(series, dtype=np.float64)
            w = np.asarray(weights, dtype=np.float64)
            if w.sum() <= 0: 
                return np.nan
            val = np.average(v, weights=w)
            # clamp tiny numeric drift
            return float(np.clip(val, -1.0, 1.0))

        layer_cos = (
            g.apply(lambda d: pd.Series({
                "mean_cos_base_ft": _wavg_cos(d["cos_base_ft"], d["n_params"])
            }))
            .reset_index()
            .sort_values(["bucket","layer_idx"])
        )
        layer_cos["one_minus_cos"] = 1.0 - layer_cos["mean_cos_base_ft"]
        layer_cos.to_csv(output_path / "weight_cosine_per_layer_component__weightedMean.csv", index=False)

        for metric, fname in [("mean_cos_base_ft", "plot_weight_mean_cosine_per_layer.png"),
                            ("one_minus_cos",   "plot_weight_one_minus_cos_per_layer.png")]:
            plt.figure()
            for bname in sorted(layer_cos["bucket"].unique()):
                sub = layer_cos[layer_cos["bucket"] == bname].sort_values("layer_idx")
                plt.plot(sub["layer_idx"], sub[metric], label=bname)
            plt.xlabel("Layer")
            plt.ylabel(metric.replace("_", " "))
            plt.title("Weights (param-weighted): " + metric.replace("_", " "))
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path / fname, dpi=180)
            plt.close()

    # Top-k per component
    topk_rows = []
    for b in sorted(df["bucket"].unique()):
        sub = df[df["bucket"] == b].sort_values("delta_fro", ascending=False).head(topk)
        for _, r in sub.iterrows():
            topk_rows.append({"bucket": b, "param_name": r["param_name"], "layer_idx": int(r["layer_idx"]), "delta_fro": float(r["delta_fro"]) })
    if topk_rows:
        pd.DataFrame(topk_rows).to_csv(output_path / "weight_deltas_topk_per_bucket.csv", index=False)

    # Plots
    if not layer_comp_pivot.empty:
        fig, ax = plt.subplots(figsize=(16, 9))
        for comp in layer_comp_pivot.columns:
            ax.plot(layer_comp_pivot.index, layer_comp_pivot[comp].values, label=comp, lw=2)
        ax.set_title("||ΔW||₍F₎ per Layer by Component", fontsize=16, weight='bold')
        ax.set_xlabel("Decoder Layer"); ax.set_ylabel("Frobenius Norm of ΔW")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5); ax.legend(ncol=2)
        plt.tight_layout(); fig.savefig(output_path / "analysis_weight_deltas_layer_component.png"); plt.close(fig)

    agg_layer = df[df["layer_idx"] >= 0].groupby("layer_idx")["delta_fro"].sum().reset_index()
    agg_bucket = df.groupby(["bucket"])["delta_fro"].sum().reset_index().sort_values("delta_fro", ascending=False)
    agg_bucket_normed = df.groupby(["bucket"])["delta_per_param"].mean().reset_index().sort_values("delta_per_param", ascending=False)

    agg_layer.to_csv(output_path / "weight_deltas_per_layer.csv", index=False)
    agg_bucket.to_csv(output_path / "weight_deltas_per_bucket_sum.csv", index=False)
    agg_bucket_normed.to_csv(output_path / "weight_deltas_per_bucket_normalized.csv", index=False)

    if not agg_layer.empty:
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.bar(agg_layer["layer_idx"], agg_layer["delta_fro"], color='indigo', alpha=0.8)
        ax.set_title('Fine-Tuning Impact: L2 Norm of Weight Deltas per Layer', fontsize=16, weight='bold')
        ax.set_xlabel('Decoder Layer'); ax.set_ylabel('Sum of L2 Norms of Parameter Deltas')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        fig.savefig(output_path / "analysis_weight_deltas.png")
        plt.close(fig)

    if not agg_bucket.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(agg_bucket["bucket"], agg_bucket["delta_fro"], alpha=0.85)
        ax.set_title("ΔW by Module Bucket (sum across layers)", fontsize=14, weight='bold')
        ax.set_xlabel("Σ_layers ||ΔW||_F"); ax.set_ylabel("Module")
        plt.tight_layout()
        fig.savefig(output_path / "analysis_weight_deltas_by_bucket.png")
        plt.close(fig)

    # Heatmap of ΔW by layer × component
    layer_comp_pivot = df.pivot_table(index="layer_idx", columns="bucket", values="delta_fro", aggfunc="sum")
    if not layer_comp_pivot.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(layer_comp_pivot.T, cmap="viridis", annot=False,
                    cbar_kws={'label': '||ΔW||₍F₎'})
        ax.set_title("Weight ΔW by Layer × Component", fontsize=14, weight='bold')
        ax.set_xlabel("Layer"); ax.set_ylabel("Component")
        plt.tight_layout()
        fig.savefig(output_path / "weight_deltas_layer_component_heatmap.png")
        plt.close(fig)

    # heatmap of RELATIVE change (||ΔW|| / ||W_base||) by layer × component
    # Uses the pivot you already computed above (pivot_rel_delta)
    rel_heat = pivot_rel_delta.T  # components as rows, layers as columns
    if not rel_heat.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(rel_heat, cmap="viridis", cbar_kws={'label': '||ΔW||₍F₎ / ||W_base||₍F₎'})
        ax.set_title("Relative Weight Change by Component × Layer")
        ax.set_xlabel("Layer"); ax.set_ylabel("Component")
        plt.tight_layout(); fig.savefig(output_path / "weights_relative_change_heatmap.png", dpi=180); plt.close(fig)

# PII
@torch.no_grad()
def compute_pii_for_item(model: PreTrainedModel, tokenizer: AutoTokenizer,
                         instruction: str, paraphrase: str, unrelated_instr: str,
                         inp: str, device: str,
                         focus_on_answer: bool = False,
                         gen_kwargs: Optional[dict] = None,
                         cache: Optional[AnswerCache] = None,
                         model_id: str = "") -> Dict[int, float]:
    """
    PII_l = 1 - (cos(orig, paraphrase) / cos(orig, unrelated)), using masked-mean hidden states per layer.
    If focus_on_answer, we generate answers for each input and restrict masking to seg_answer.
    """
    gen_kwargs = gen_kwargs or {}
    ans_o = ans_p = ans_u = None
    if focus_on_answer:
        ans_o = generate_answer_text(model, tokenizer, instruction, inp, device, cache=cache, model_id=model_id, **gen_kwargs)
        ans_p = generate_answer_text(model, tokenizer, paraphrase,  inp, device, cache=cache, model_id=model_id, **gen_kwargs)
        ans_u = generate_answer_text(model, tokenizer, unrelated_instr, inp, device, cache=cache, model_id=model_id, **gen_kwargs)

    enc_o = tokenise_with_segments(tokenizer, instruction, inp, device, answer_text=ans_o if focus_on_answer else None)
    enc_p = tokenise_with_segments(tokenizer, paraphrase,  inp, device, answer_text=ans_p if focus_on_answer else None)
    enc_u = tokenise_with_segments(tokenizer, unrelated_instr, inp, device, answer_text=ans_u if focus_on_answer else None)

    with ActivationExtractor(model) as ex:
        _ = model(enc_o["input_ids"], attention_mask=enc_o["attention_mask"], use_cache=True)
        h_o = dict(sorted(ex.hidden_states.items()))
    with ActivationExtractor(model) as ex:
        _ = model(enc_p["input_ids"], attention_mask=enc_p["attention_mask"], use_cache=True)
        h_p = dict(sorted(ex.hidden_states.items()))
    with ActivationExtractor(model) as ex:
        _ = model(enc_u["input_ids"], attention_mask=enc_u["attention_mask"], use_cache=True)
        h_u = dict(sorted(ex.hidden_states.items()))

    if focus_on_answer and enc_o["seg_answer"].any():
        mo = enc_o["seg_answer"]; mp = enc_p["seg_answer"]; mu = enc_u["seg_answer"]
    else:
        mo = enc_o["attention_mask"]; mp = enc_p["attention_mask"]; mu = enc_u["attention_mask"]

    pii = {}
    for l in h_o.keys():
        o = masked_mean(h_o[l][0], mo[0].bool())
        p = masked_mean(h_p[l][0], mp[0].bool())
        u = masked_mean(h_u[l][0], mu[0].bool())
        cos_op = torch.nn.functional.cosine_similarity(o.view(1,-1), p.view(1,-1)).item()
        cos_ou = torch.nn.functional.cosine_similarity(o.view(1,-1), u.view(1,-1)).item()
        value = 1.0 - (cos_op / max(1e-6, cos_ou))
        pii[l] = value
    return pii

# Model Loading
#def load_model_and_tokenizer(model_path: str, base_model_path: str, device: str, attn_impl: Optional[str] = None, dtype_str: str = "bfloat16") -> tuple[PreTrainedModel, AutoTokenizer, str]:
def load_model_and_tokenizer(model_path, base_model_path, device, attn_impl=None, dtype_str="bfloat16",
                             merge_lora_for_weights=False):
    path, base_path = Path(model_path), Path(base_model_path)
    if not base_path.exists(): script_logger.error(f"Base model path not found: {base_path}"); sys.exit(1)
    script_logger.info(f"Loading tokenizer from: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16 if dtype_str == "float16" else torch.float32

    def _apply_attn_impl(model):
        # Best-effort apply attention implementation
        if attn_impl:
            try:
                model.config.attn_implementation = attn_impl
                script_logger.info(f"Set attention implementation to: {attn_impl}")
            except Exception:
                pass
        return model

    model_id_for_cache = str(path)

    if (path / "adapter_config.json").exists() and PeftModel is not None:
        script_logger.info("Detected LoRA adapters. Loading base + applying adapters.")
        base = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch_dtype, device_map={"": device})
        base = _apply_attn_impl(base)
        peft = PeftModel.from_pretrained(base, model_path)

        # optionally merge adapters so named_parameters() reflect effective weights
        if merge_lora_for_weights:
            script_logger.info("Merging LoRA adapters into the base weights for analysis…")
            try:
                peft = peft.merge_and_unload()
                script_logger.info("Merge successful; proceeding with merged effective weights.")
            except Exception as e:
                script_logger.warning(f"Merge failed, continuing without merge (will not see ΔW in raw weights): {e}")

        model = peft
    else:
        script_logger.info("Detected a full model. Loading directly.")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, device_map={"": device})
        model = _apply_attn_impl(model)

    try:
        model.config.use_cache = True
    except Exception:
        pass

    if model is None: script_logger.error(f"Could not load model from {model_path}"); sys.exit(1)
    model.eval()
    return model, tokenizer, model_id_for_cache

def get_group_wise_split_ids(all_prompt_ids: List[int], val_size: float, test_size: float, seed: int) -> Tuple[Set[int], Set[int], Set[int]]:
    prompt_ids = sorted(list(set(all_prompt_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(prompt_ids)
    n_test, n_val = int(len(prompt_ids) * test_size), int(len(prompt_ids) * val_size)
    test_ids, val_ids = set(prompt_ids[:n_test]), set(prompt_ids[n_test : n_test + n_val])
    train_ids = set(prompt_ids[n_test + n_val:])
    script_logger.info(f"Group-wise split complete. Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    return train_ids, val_ids, test_ids

# Aggregate plotting/reporting (kept, minor tweaks)
def plot_aggregate_results(agg_data: Dict, num_samples: int, output_path: Path):
    if not any(agg_data.values()):
        script_logger.warning("No data for aggregation plot. Skipping.")
        return

    sim_base_records = [(pkey, layer, val) for pkey, layers in agg_data['base_sims'].items() for layer, vals in layers.items() for val in vals]
    df_sim_base = pd.DataFrame(sim_base_records, columns=['paraphrase', 'layer', 'value'])
    sim_ft_records = [(pkey, layer, val) for pkey, layers in agg_data['ft_sims'].items() for layer, vals in layers.items() for val in vals]
    df_sim_ft = pd.DataFrame(sim_ft_records, columns=['paraphrase', 'layer', 'value'])
    df_sim_delta = df_sim_ft['value'] - df_sim_base['value']
    df_sim_delta = pd.concat([df_sim_base[['paraphrase', 'layer']], df_sim_delta.rename('delta')], axis=1)

    l2_base_records = [(pkey, layer, val) for pkey, layers in agg_data['base_l2'].items() for layer, vals in layers.items() for val in vals]
    df_l2_base = pd.DataFrame(l2_base_records, columns=['paraphrase', 'layer', 'value'])
    l2_ft_records = [(pkey, layer, val) for pkey, layers in agg_data['ft_l2'].items() for layer, vals in layers.items() for val in vals]
    df_l2_ft = pd.DataFrame(l2_ft_records, columns=['paraphrase', 'layer', 'value'])
    df_l2_delta = None
    if not df_l2_base.empty and not df_l2_ft.empty:
        df_l2_delta = pd.concat([df_l2_base[['paraphrase','layer']],
                                 (df_l2_ft['value'] - df_l2_base['value']).rename('delta')], axis=1)

    norm_base_records = [(pkey, layer, val) for pkey, layers in agg_data['base_norms'].items() for layer, vals in layers.items() for val in vals]
    df_norm_base = pd.DataFrame(norm_base_records, columns=['paraphrase', 'layer', 'value'])
    norm_ft_records = [(pkey, layer, val) for pkey, layers in agg_data['ft_norms'].items() for layer, vals in layers.items() for val in vals]
    df_norm_ft = pd.DataFrame(norm_ft_records, columns=['paraphrase', 'layer', 'value'])
    df_norm_delta = df_norm_ft['value'] - df_norm_base['value']
    df_norm_delta = pd.concat([df_norm_base[['paraphrase', 'layer']], df_norm_delta.rename('delta')], axis=1)

    emb_base_records = [(pkey, val) for pkey, vals in agg_data['base_emb_sims'].items() for val in vals]
    df_emb_base = pd.DataFrame(emb_base_records, columns=['paraphrase', 'value'])
    emb_ft_records = [(pkey, val) for pkey, vals in agg_data['ft_emb_sims'].items() for val in vals]
    df_emb_ft = pd.DataFrame(emb_ft_records, columns=['paraphrase', 'value'])
    df_emb_delta = df_emb_ft['value'] - df_emb_base['value']
    df_emb_delta = pd.concat([df_emb_base[['paraphrase']], df_emb_delta.rename('delta')], axis=1)

    if not df_sim_delta.empty:
        mean_by_layer = df_sim_delta.groupby('layer')['delta'].mean()
        sem_by_layer = df_sim_delta.groupby('layer')['delta'].sem()
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(mean_by_layer.index, mean_by_layer.values, 'o-', color='mediumseagreen', label='Mean Δ Similarity')
        ax.fill_between(mean_by_layer.index, mean_by_layer.values - 1.96 * sem_by_layer, mean_by_layer.values + 1.96 * sem_by_layer, color='mediumseagreen', alpha=0.2, label='95% CI')
        ax.axhline(0, color='black', lw=1, linestyle='--'); ax.legend(); ax.grid(True)
        ax.set_title(f'Overall Average Change in Representational Similarity (N={num_samples})', fontsize=16, weight='bold')
        ax.set_xlabel('Decoder Layer'); ax.set_ylabel('Average Δ Cosine Similarity (FT - Base)')
        plt.tight_layout(); fig.savefig(output_path / "aggregate_lineplot_sim_delta.png"); plt.close(fig)

    if df_l2_delta is not None and not df_l2_delta.empty:
        mean_by_layer = df_l2_delta.groupby('layer')['delta'].mean()
        sem_by_layer  = df_l2_delta.groupby('layer')['delta'].sem()
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(mean_by_layer.index, mean_by_layer.values, 'o-', label='Mean Δ L2 (FT − Base)')
        ax.fill_between(mean_by_layer.index,
                        mean_by_layer.values - 1.96 * sem_by_layer,
                        mean_by_layer.values + 1.96 * sem_by_layer,
                        alpha=0.2, label='95% CI')
        ax.axhline(0, color='black', lw=1, linestyle='--'); ax.legend(); ax.grid(True)
        ax.set_title(f'Pairwise Distance Shrinkage (orig↔para) by Layer (N={num_samples})', fontsize=16, weight='bold')
        ax.set_xlabel('Decoder Layer'); ax.set_ylabel('Δ L2 (FT − Base)')
        plt.tight_layout(); fig.savefig(output_path / "aggregate_lineplot_l2_shrinkage.png"); plt.close(fig)

    if not df_sim_delta.empty:
        heatmap_data = df_sim_delta.groupby(['paraphrase', 'layer'])['delta'].mean().unstack(level='layer')
        fig, ax = plt.subplots(figsize=(16, max(8, len(heatmap_data.index) * 0.5)))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", ax=ax, cbar_kws={'label': 'Average Δ Cosine Similarity (FT - Base)'})
        ax.set_title(f'Aggregate: Mean Change in Similarity by Paraphrase (N={num_samples})', fontsize=16, weight='bold')
        ax.set_xlabel('Decoder Layer'); ax.set_ylabel('Paraphrase Type')
        plt.xticks(rotation=45); plt.tight_layout(); fig.savefig(output_path / "aggregate_heatmap_sim_delta.png"); plt.close(fig)

    if not df_norm_delta.empty:
        mean_by_layer = df_norm_delta.groupby('layer')['delta'].mean()
        sem_by_layer = df_norm_delta.groupby('layer')['delta'].sem()
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(mean_by_layer.index, mean_by_layer.values, 'o-', color='darkslateblue', label='Mean Δ Activation Norm')
        ax.fill_between(mean_by_layer.index, mean_by_layer.values - 1.96 * sem_by_layer, mean_by_layer.values + 1.96 * sem_by_layer, color='darkslateblue', alpha=0.2, label='95% CI')
        ax.axhline(0, color='black', lw=1, linestyle='--'); ax.legend(); ax.grid(True)
        ax.set_title(f'Overall Average Change in Activation Norm (N={num_samples})', fontsize=16, weight='bold')
        ax.set_xlabel('Decoder Layer'); ax.set_ylabel('Average Δ Mean L2 Norm (FT - Base)')
        plt.tight_layout(); fig.savefig(output_path / "aggregate_lineplot_norm_delta.png"); plt.close(fig)

    if not df_emb_delta.empty:
        fig, ax = plt.subplots(figsize=(14, max(8, len(df_emb_delta['paraphrase'].unique()) * 0.6)))
        sns.boxplot(data=df_emb_delta, x='delta', y='paraphrase', orient='h', ax=ax, )
        ax.axvline(0, color='black', lw=1, linestyle='--')
        ax.set_title(f'Change in Embedding Space Similarity After Fine-Tuning (N={num_samples})', fontsize=16, weight='bold')
        ax.set_xlabel('Δ Cosine Similarity (FT - Base)'); ax.set_ylabel('Paraphrase Type')
        ax.grid(True, axis='x'); plt.tight_layout(); fig.savefig(output_path / "aggregate_boxplot_embedding_delta.png"); plt.close(fig)
    script_logger.info("Saved all aggregate plots.")

# Scores + Correlations
def load_scores(scores_path: str) -> Dict[Tuple[int, str], float]:
    if not scores_path: return {}
    with open(scores_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for item in data:
        pid = item.get("prompt_count")
        if pid is None: continue
        for k, v in item.items():
            if k.startswith("instruct_") or k == "instruction_original":
                if isinstance(v, list) and len(v) >= 1:
                    mapping[(pid, k)] = float(v[0])
    return mapping

def compute_correlations(item_rows: List[Dict[str, Any]], scores_base: Dict, scores_ft: Dict, output_path: Path):
    if not scores_base or not scores_ft:
        script_logger.info("Scores not provided for correlations; skipping.")
        return
    records = []
    for row in item_rows:
        pid, pkey = row["prompt_count"], row["paraphrase_key"]
        if (pid, pkey) in scores_base and (pid, pkey) in scores_ft:
            delta_tf = scores_ft[(pid, pkey)] - scores_base[(pid, pkey)]
            rec = {"prompt_count": pid, "paraphrase_key": pkey, "delta_tf": delta_tf}
            rec.update(row)
            records.append(rec)
    if not records:
        script_logger.warning("No overlapping score entries for correlation.")
        return
    df = pd.DataFrame(records)
    metric_cols = [c for c in df.columns if (c.startswith("delta_") or c.startswith("diff_of_diffs_")) and c not in {"delta_tf"}]
    corr_rows = []
    for m in metric_cols:
        x = df[m].values
        y = df["delta_tf"].values
        if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
            pr, pp = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
            corr_rows.append({"metric": m, "pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp,
                              "n": len(x), "x_mean": float(np.mean(x)), "x_std": float(np.std(x)), "y_mean": float(np.mean(y)), "y_std": float(np.std(y))})
    cdf = pd.DataFrame(corr_rows).sort_values("pearson_r", ascending=False)
    cdf.to_csv(output_path / "correlations_metrics_vs_deltaTF.csv", index=False)
    df.to_csv(output_path / "per_item_metrics_with_deltaTF.csv", index=False)
    script_logger.info("Saved correlations with ΔTF.")

# Experiments
@torch.no_grad()
def run_best_worst_experiment(base_model: PreTrainedModel, ft_model: PreTrainedModel, tokenizer: AutoTokenizer,
                              prompts_data: List[Dict[str, Any]], scores_base: Dict[Tuple[int, str], float], scores_ft, device: str, output_path: Path,
                              gen_kwargs: dict, cache: AnswerCache, base_model_id: str, ft_model_id: str):
    rows = []
    for item in tqdm(prompts_data, desc="Best-vs-Worst", leave=False):
        pid = item.get('prompt_count')
        if pid is None: continue
        pkeys = [k for k in item.keys() if (k.startswith('instruct_') or k == 'instruction_original') and (pid, k) in scores_base and item.get(k)]
        if len(pkeys) < 2: continue
        scores_here = [(k, scores_base[(pid, k)]) for k in pkeys]
        best_k, _ = max(scores_here, key=lambda kv: kv[1])
        worst_k, _ = min(scores_here, key=lambda kv: kv[1])

        instr_best  = item[best_k] if best_k != "instruction_original" else item["instruction_original"]
        instr_worst = item[worst_k] if worst_k != "instruction_original" else item["instruction_original"]
        inp = item.get('input', '')

        ans_best_base  = generate_answer_text(base_model, tokenizer, instr_best,  inp, device, cache=cache, model_id=base_model_id, **gen_kwargs)
        ans_worst_base = generate_answer_text(base_model, tokenizer, instr_worst, inp, device, cache=cache, model_id=base_model_id, **gen_kwargs)
        ans_best_ft    = generate_answer_text(ft_model,   tokenizer, instr_best,  inp, device, cache=cache, model_id=ft_model_id, **gen_kwargs)
        ans_worst_ft   = generate_answer_text(ft_model,   tokenizer, instr_worst, inp, device, cache=cache, model_id=ft_model_id, **gen_kwargs)

        enc_o_base, enc_p_base = encode_pair(tokenizer, instr_best, instr_worst, inp, device, ans_best_base, ans_worst_base)
        enc_o_ft,   enc_p_ft   = encode_pair(tokenizer, instr_best, instr_worst, inp, device, ans_best_ft,   ans_worst_ft)

        # collate (batch=2)
        batch_base = collate_batch([enc_o_base, enc_p_base], device)
        batch_ft   = collate_batch([enc_o_ft, enc_p_ft], device)

        with ActivationExtractor(base_model) as ex:
            _ = base_model(batch_base.input_ids, attention_mask=batch_base.attention_mask, use_cache=True)
            hs_base = dict(sorted(ex.hidden_states.items()))
        with ActivationExtractor(ft_model) as ex:
            _ = ft_model(batch_ft.input_ids, attention_mask=batch_ft.attention_mask, use_cache=True)
            hs_ft = dict(sorted(ex.hidden_states.items()))

        mb1 = batch_base.seg_answer; mb2 = batch_base.seg_answer
        mf1 = batch_ft.seg_answer;   mf2 = batch_ft.seg_answer

        l2_base = l2_by_layer_masked(hs_base, 0, 1, mb1, mb2)
        l2_ft   = l2_by_layer_masked(hs_ft,   0, 1, mf1, mf2)
        final_layer = max(l2_base.keys())
        rows.append({
            "prompt_count": pid,
            "best_key": best_k, "worst_key": worst_k,
            "l2_base_final": l2_base[final_layer],
            "l2_ft_final": l2_ft[final_layer],
            "delta_l2_final": l2_ft[final_layer] - l2_base[final_layer]
        })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path / "best_worst_answer_distance.csv", index=False)
        gmean_base = df["l2_base_final"].mean(); gmean_ft = df["l2_ft_final"].mean()
        shrink = (df["delta_l2_final"].mean(), df["delta_l2_final"].median())
        summary = [
            "# Best-vs-Worst Answer Distance (final layer)",
            f"- N prompts: {len(df)}",
            f"- Mean L2 (Base): {gmean_base:.6f}",
            f"- Mean L2 (FT):   {gmean_ft:.6f}",
            f"- Mean Δ L2 (FT−Base): {shrink[0]:+.6f}  (median {shrink[1]:+.6f})",
        ]
        (output_path / "best_worst_answer_distance.md").write_text("\n".join(summary), encoding="utf-8")

    # Optional: correlate TF gap with distances
    if scores_base and scores_ft:
        try:
            from scipy import stats
            tf_gaps, l2_base, l2_ft = [], [], []
            for _, row in df.iterrows():
                pid = row["prompt_count"]
                best_key, worst_key = row["best_key"], row["worst_key"]
                tf_best = scores_base.get((pid, best_key), None)
                tf_worst = scores_base.get((pid, worst_key), None)
                if tf_best is not None and tf_worst is not None:
                    tf_gaps.append(tf_best - tf_worst)
                    l2_base.append(row["l2_base_final"])
                    l2_ft.append(row["l2_ft_final"])
            if tf_gaps:
                rho_b, p_b = stats.spearmanr(tf_gaps, l2_base)
                rho_f, p_f = stats.spearmanr(tf_gaps, l2_ft)
                with open(output_path / "best_worst_correlation.md", "w") as f:
                    f.write(f"Spearman correlation (Base): rho={rho_b:.3f}, p={p_b:.3g}\n")
                    f.write(f"Spearman correlation (FT):   rho={rho_f:.3f}, p={p_f:.3g}\n")
        except Exception as e:
            script_logger.warning(f"Best-vs-worst correlation failed: {e}")

# Batched aggregate runner
@torch.no_grad()
def run_aggregate(
    base_model: PreTrainedModel,
    ft_model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompts_data: List[Dict[str, Any]],
    device: str,
    output_path: Path,
    args,
    base_model_id: str,
    ft_model_id: str,
):
    # Detailed ΔW (component/top-k; faster; tqdm)
    analyze_and_plot_weight_deltas(
        base_model, ft_model, output_path,
        compute_svd=args.compute_svd,
        topk=args.topk_weights,
        fast_mode=True,
        include_buckets=None,  # or pass a list to narrow
        args=args
    )

    aggregate_data = {
        'base_emb_sims': defaultdict(list), 'ft_emb_sims': defaultdict(list),
        'base_sims': defaultdict(lambda: defaultdict(list)), 'ft_sims': defaultdict(lambda: defaultdict(list)),
        'base_l2': defaultdict(lambda: defaultdict(list)),  'ft_l2': defaultdict(lambda: defaultdict(list)),
        'base_norms': defaultdict(lambda: defaultdict(list)), 'ft_norms': defaultdict(lambda: defaultdict(list))
    }
    seg_norm_base, seg_norm_ft = defaultdict(lambda: defaultdict(lambda: defaultdict(list))), defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seg_cos_base, seg_cos_ft = defaultdict(lambda: defaultdict(lambda: defaultdict(list))), defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    pii_base_per_layer, pii_ft_per_layer = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    per_item_rows_for_corr = []
    dod_rows = []
    outlier_records = []
    emb_sim_deltas_for_mad = []

    # Precompute unrelated mapping
    originals = [(it['prompt_count'], it.get('instruction_original', '')) for it in prompts_data if it.get('instruction_original')]
    unrelated_lookup = {}
    for i, (pid, instr) in enumerate(originals):
        j = (i + 1) % len(originals) if originals else 0
        if originals:
            unrelated_lookup[pid] = originals[j][1]

    # Build all (pid, pkey) pairs
    pairs = []
    for item in prompts_data:
        pid = item.get('prompt_count')
        if pid is None: continue
        original_text = item.get('instruction_original')
        inp = item.get('input', '')
        if not original_text: continue
        keys_to_process = args.paraphrase_keys if args.paraphrase_keys else [k for k in item if k.startswith('instruct_') and item[k]]
        for p_key in keys_to_process:
            paraphrase_text = item.get(p_key)
            if not paraphrase_text: continue
            pairs.append((pid, p_key, original_text, paraphrase_text, inp))

    total = len(pairs)
    if total == 0:
        script_logger.warning("No pairs to process.")
        return

    # Answer cache and gen kwargs
    cache = AnswerCache() if args.focus_on_answer_tokens or args.pii_focus_on_answer else None
    gen_kwargs = dict(max_new_tokens=args.gen_max_new_tokens, temperature=args.gen_temperature, top_p=args.gen_top_p)

    # Progress bar over batches
    bs = max(1, args.pairs_per_batch)
    with tqdm(total=total, desc="Pairs", unit="pair") as pbar:
        for start in range(0, total, bs):
            batch_pairs = pairs[start:start+bs]

            # Prepare/generate answers if needed
            ans_map_base = {}
            ans_map_ft   = {}
            if args.focus_on_answer_tokens:
                for (pid, pkey, orig, para, inp) in batch_pairs:
                    ans_map_base[(orig, inp)] = generate_answer_text(base_model, tokenizer, orig, inp, device, cache=cache, model_id=base_model_id, **gen_kwargs)
                    ans_map_base[(para, inp)] = generate_answer_text(base_model, tokenizer, para, inp, device, cache=cache, model_id=base_model_id, **gen_kwargs)
                    ans_map_ft[(orig, inp)]   = generate_answer_text(ft_model,   tokenizer, orig, inp, device, cache=cache, model_id=ft_model_id, **gen_kwargs)
                    ans_map_ft[(para, inp)]   = generate_answer_text(ft_model,   tokenizer, para, inp, device, cache=cache, model_id=ft_model_id, **gen_kwargs)

            # Encode all examples for base and ft (flattened as [o1, p1, o2, p2, ...])
            base_examples: List[EncodedExample] = []
            ft_examples:   List[EncodedExample] = []
            index_map = []  # maps pair index -> (orig_idx, para_idx)
            for (pid, pkey, orig, para, inp) in batch_pairs:
                ans_ob = ans_map_base.get((orig, inp), None) if args.focus_on_answer_tokens else None
                ans_pb = ans_map_base.get((para, inp), None) if args.focus_on_answer_tokens else None
                ans_of = ans_map_ft.get((orig, inp),   None) if args.focus_on_answer_tokens else None
                ans_pf = ans_map_ft.get((para, inp),   None) if args.focus_on_answer_tokens else None

                enc_ob, enc_pb = encode_pair(tokenizer, orig, para, inp, device, ans_ob, ans_pb)
                enc_of, enc_pf = encode_pair(tokenizer, orig, para, inp, device, ans_of, ans_pf)

                base_examples.extend([enc_ob, enc_pb])
                ft_examples.extend([enc_of, enc_pf])
                index_map.append((len(base_examples)-2, len(base_examples)-1))  # indices in the flattened batch

            # Collate
            batch_base = collate_batch(base_examples, device)
            batch_ft   = collate_batch(ft_examples, device)

            # Decide mask to use (attention vs seg_answer)
            def pick_mask(batch_enc):
                if args.focus_on_answer_tokens and batch_enc.seg_answer.any():
                    return batch_enc.seg_answer
                return batch_enc.attention_mask

            mask_base = pick_mask(batch_base)
            mask_ft   = pick_mask(batch_ft)

            # Forward (hidden states) + optional component probes
            with ActivationExtractor(base_model) as exb, ComponentProbe(
                base_model,
                capture_attn_o=args.capture_component_activations,
                capture_mlp_down=args.capture_component_activations,
                capture_attn_qkv=args.capture_attn_qkv,
                capture_mlp_up_gate=args.capture_mlp_up_gate,
            ) as pb:
                _ = base_model(batch_base.input_ids, attention_mask=batch_base.attention_mask,
                   use_cache=True)
                hs_base = dict(sorted(exb.hidden_states.items()))
                comp_attn_base = dict(sorted(pb.attn_o.items()))
                comp_mlp_base  = dict(sorted(pb.mlp_down.items()))
            with ActivationExtractor(ft_model) as exf, ComponentProbe(
                ft_model,
                capture_attn_o=args.capture_component_activations,
                capture_mlp_down=args.capture_component_activations,
                capture_attn_qkv=args.capture_attn_qkv,
                capture_mlp_up_gate=args.capture_mlp_up_gate,
            ) as pf:
                _ = ft_model(batch_ft.input_ids, attention_mask=batch_ft.attention_mask,
                   use_cache=True)
                hs_ft = dict(sorted(exf.hidden_states.items()))
                comp_attn_ft = dict(sorted(pf.attn_o.items()))
                comp_mlp_ft  = dict(sorted(pf.mlp_down.items()))

            # Embeddings (masked means on whole prompt or answers included if provided)
            emb = base_model.get_input_embeddings()
            emb_base = emb(batch_base.input_ids)
            emb_ft   = ft_model.get_input_embeddings()(batch_ft.input_ids)

            # For each pair in the batch, compute metrics
            for bp_idx, (orig_idx, para_idx) in enumerate(index_map):
                pid, p_key, orig, para, inp = batch_pairs[bp_idx]

                # Embedding similarity (mean over chosen mask = attention or answer)
                eo = masked_mean(emb_base[orig_idx], mask_base[orig_idx].bool())
                ep = masked_mean(emb_base[para_idx], mask_base[para_idx].bool())
                base_emb_sim = cosine(eo, ep, upcast_fp32=args.cosine_upcast_fp32)

                fo = masked_mean(emb_ft[orig_idx],   mask_ft[orig_idx].bool())
                fp = masked_mean(emb_ft[para_idx],   mask_ft[para_idx].bool())
                ft_emb_sim = cosine(fo, fp, upcast_fp32=args.cosine_upcast_fp32)

                # Cosine + L2 per layer
                base_cos = cosine_by_layer_masked(hs_base, orig_idx, para_idx, mask_base, mask_base)
                ft_cos   = cosine_by_layer_masked(hs_ft,   orig_idx, para_idx, mask_ft,   mask_ft)
                base_l2  = l2_by_layer_masked(hs_base, orig_idx, para_idx, mask_base, mask_base)
                ft_l2    = l2_by_layer_masked(hs_ft,   orig_idx, para_idx, mask_ft,   mask_ft)

                # Norms on paraphrase
                base_norms_para = norms_by_layer_masked(hs_base, para_idx, mask_base)
                ft_norms_para   = norms_by_layer_masked(hs_ft,   para_idx, mask_ft)

                # Component activations (optional)
                if args.capture_component_activations:
                    comp_attn_base_norms = component_norms(comp_attn_base, para_idx, mask_base)
                    comp_mlp_base_norms  = component_norms(comp_mlp_base,  para_idx, mask_base)
                    comp_attn_ft_norms   = component_norms(comp_attn_ft,   para_idx, mask_ft)
                    comp_mlp_ft_norms    = component_norms(comp_mlp_ft,    para_idx, mask_ft)

                    # component-wise cosine similarities (orig ↔ para)
                    base_attn_cos = component_cosine_by_layer(comp_attn_base, orig_idx, para_idx, mask_base, mask_base)
                    base_mlp_cos  = component_cosine_by_layer(comp_mlp_base,  orig_idx, para_idx, mask_base, mask_base)
                    ft_attn_cos   = component_cosine_by_layer(comp_attn_ft,   orig_idx, para_idx, mask_ft,   mask_ft)
                    ft_mlp_cos    = component_cosine_by_layer(comp_mlp_ft,    orig_idx, para_idx, mask_ft,   mask_ft)

                    for l, v in base_attn_cos.items():
                        seg_cos_base[p_key]["attn_o_proj"][l].append(v)
                    for l, v in base_mlp_cos.items():
                        seg_cos_base[p_key]["mlp_down_proj"][l].append(v)
                    for l, v in ft_attn_cos.items():
                        seg_cos_ft[p_key]["attn_o_proj"][l].append(v)
                    for l, v in ft_mlp_cos.items():
                        seg_cos_ft[p_key]["mlp_down_proj"][l].append(v)

                    # Optional: if also capture q/k/v and mlp up/gate
                    if args.capture_attn_qkv:
                        for l, v in component_cosine_by_layer(pb.attn_q, orig_idx, para_idx, mask_base, mask_base).items():
                            seg_cos_base[p_key]["attn_q_proj"][l].append(v)
                        for l, v in component_cosine_by_layer(pb.attn_k, orig_idx, para_idx, mask_base, mask_base).items():
                            seg_cos_base[p_key]["attn_k_proj"][l].append(v)
                        for l, v in component_cosine_by_layer(pb.attn_v, orig_idx, para_idx, mask_base, mask_base).items():
                            seg_cos_base[p_key]["attn_v_proj"][l].append(v)
                        for l, v in component_cosine_by_layer(pf.attn_q, orig_idx, para_idx, mask_ft, mask_ft).items():
                            seg_cos_ft[p_key]["attn_q_proj"][l].append(v)
                        for l, v in component_cosine_by_layer(pf.attn_k, orig_idx, para_idx, mask_ft, mask_ft).items():
                            seg_cos_ft[p_key]["attn_k_proj"][l].append(v)
                        for l, v in component_cosine_by_layer(pf.attn_v, orig_idx, para_idx, mask_ft, mask_ft).items():
                            seg_cos_ft[p_key]["attn_v_proj"][l].append(v)

                    if args.capture_mlp_up_gate:
                        for l, v in component_cosine_by_layer(pb.mlp_up, orig_idx, para_idx, mask_base, mask_base).items():
                            seg_cos_base[p_key]["mlp_up_proj"][l].append(v)
                        for l, v in component_cosine_by_layer(pb.mlp_gate, orig_idx, para_idx, mask_base, mask_base).items():
                            seg_cos_base[p_key]["mlp_gate_proj"][l].append(v)
                        for l, v in component_cosine_by_layer(pf.mlp_up, orig_idx, para_idx, mask_ft, mask_ft).items():
                            seg_cos_ft[p_key]["mlp_up_proj"][l].append(v)
                        for l, v in component_cosine_by_layer(pf.mlp_gate, orig_idx, para_idx, mask_ft, mask_ft).items():
                            seg_cos_ft[p_key]["mlp_gate_proj"][l].append(v)

                    # Save to rows (lazy store; then dump CSV per batch)
                    # We'll append to CSV after the loop for simplicity
                    # Accumulate in dicts keyed by paraphrase type
                    for l, v in comp_attn_base_norms.items():
                        seg_norm_base[p_key]["attn_o_proj"][l].append(v)
                    for l, v in comp_mlp_base_norms.items():
                        seg_norm_base[p_key]["mlp_down_proj"][l].append(v)
                    for l, v in comp_attn_ft_norms.items():
                        seg_norm_ft[p_key]["attn_o_proj"][l].append(v)
                    for l, v in comp_mlp_ft_norms.items():
                        seg_norm_ft[p_key]["mlp_down_proj"][l].append(v)

                    # Q/K/V projections
                    for l, v in component_norms(pb.attn_q, para_idx, mask_base).items():
                        seg_norm_base[p_key]["attn_q_proj"][l].append(v)
                    for l, v in component_norms(pb.attn_k, para_idx, mask_base).items():
                        seg_norm_base[p_key]["attn_k_proj"][l].append(v)
                    for l, v in component_norms(pb.attn_v, para_idx, mask_base).items():
                        seg_norm_base[p_key]["attn_v_proj"][l].append(v)

                    for l, v in component_norms(pf.attn_q, para_idx, mask_ft).items():
                        seg_norm_ft[p_key]["attn_q_proj"][l].append(v)
                    for l, v in component_norms(pf.attn_k, para_idx, mask_ft).items():
                        seg_norm_ft[p_key]["attn_k_proj"][l].append(v)
                    for l, v in component_norms(pf.attn_v, para_idx, mask_ft).items():
                        seg_norm_ft[p_key]["attn_v_proj"][l].append(v)

                    # MLP up/gate
                    for l, v in component_norms(pb.mlp_up, para_idx, mask_base).items():
                        seg_norm_base[p_key]["mlp_up_proj"][l].append(v)
                    for l, v in component_norms(pb.mlp_gate, para_idx, mask_base).items():
                        seg_norm_base[p_key]["mlp_gate_proj"][l].append(v)

                    for l, v in component_norms(pf.mlp_up, para_idx, mask_ft).items():
                        seg_norm_ft[p_key]["mlp_up_proj"][l].append(v)
                    for l, v in component_norms(pf.mlp_gate, para_idx, mask_ft).items():
                        seg_norm_ft[p_key]["mlp_gate_proj"][l].append(v)


                # Aggregate collections
                aggregate_data['base_emb_sims'][p_key].append(base_emb_sim)
                aggregate_data['ft_emb_sims'][p_key].append(ft_emb_sim)
                for layer_idx in base_cos.keys():
                    aggregate_data['base_sims'][p_key][layer_idx].append(base_cos[layer_idx])
                    aggregate_data['ft_sims'][p_key][layer_idx].append(ft_cos[layer_idx])
                    aggregate_data['base_l2'][p_key][layer_idx].append(base_l2[layer_idx])
                    aggregate_data['ft_l2'][p_key][layer_idx].append(ft_l2[layer_idx])
                    aggregate_data['base_norms'][p_key][layer_idx].append(base_norms_para[layer_idx])
                    aggregate_data['ft_norms'][p_key][layer_idx].append(ft_norms_para[layer_idx])

                # Outlier tracking (Δ embedding sim)
                emb_sim_delta = ft_emb_sim - base_emb_sim
                emb_sim_deltas_for_mad.append(emb_sim_delta)

                # Per-item rows for correlations
                final_layer = max(base_cos.keys())
                row = {
                    "prompt_count": pid,
                    "paraphrase_key": p_key,
                    "delta_emb_sim_all": emb_sim_delta,
                    "delta_cos_sim_final": ft_cos[final_layer] - base_cos[final_layer],
                    "delta_cos_sim_mean": float(np.mean(list(ft_cos.values()))) - float(np.mean(list(base_cos.values()))),
                    "cosine_sum_base": float(np.sum(list(base_cos.values()))),
                    "cosine_sum_ft":   float(np.sum(list(ft_cos.values()))),
                    "delta_cosine_sum": float(np.sum(list(ft_cos.values())) - np.sum(list(base_cos.values()))),
                    "delta_l2_final": ft_l2[final_layer] - base_l2[final_layer],
                    "diff_of_diffs_l2_final": base_l2[final_layer] - ft_l2[final_layer],
                    "delta_l2_mean": float(np.mean(list(ft_l2.values()))) - float(np.mean(list(base_l2.values()))),
                    "delta_norm_mean": float(np.mean(list(ft_norms_para.values()))) - float(np.mean(list(base_norms_para.values()))),
                }
                per_item_rows_for_corr.append(row)
                # Also record per-item final-layer DoD to a dedicated CSV later
                dod_rows.append({
                "prompt_count": pid,
                "paraphrase_key": p_key,
                "layer": final_layer,
                "l2_base": base_l2[final_layer],
                "l2_ft": ft_l2[final_layer],
                "diff_of_diffs_l2": base_l2[final_layer] - ft_l2[final_layer],
                })

                # Optional PII (answer-focused or not)
                if args.compute_pii:
                    unrelated_instr = unrelated_lookup.get(pid, orig)
                    pii_b = compute_pii_for_item(base_model, tokenizer, orig, para, unrelated_instr, inp, device,
                                                 focus_on_answer=args.pii_focus_on_answer,
                                                 gen_kwargs=gen_kwargs, cache=cache, model_id=base_model_id)
                    pii_f = compute_pii_for_item(ft_model,   tokenizer, orig, para, unrelated_instr, inp, device,
                                                 focus_on_answer=args.pii_focus_on_answer,
                                                 gen_kwargs=gen_kwargs, cache=cache, model_id=ft_model_id)
                    for l in pii_b: pii_base_per_layer[p_key][l].append(pii_b[l])
                    for l in pii_f: pii_ft_per_layer[p_key][l].append(pii_f[l])

            pbar.update(len(batch_pairs))

    # Robust outliers (MAD z)
    if emb_sim_deltas_for_mad:
        arr = np.array(emb_sim_deltas_for_mad)
        z = robust_mad_z(arr)
        THRESH = 3.5
        z_idx = 0
        for (pid, p_key, *_rest) in pairs:
            if z_idx >= len(z): break
            if abs(z[z_idx]) >= THRESH:
                outlier_records.append({"pid": pid, "p_key": p_key, "mad_z": float(z[z_idx])})
            z_idx += 1

    # Write auxiliary CSVs
    if outlier_records:
        report_path = output_path / "embedding_similarity_outliers_mad.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Embedding Similarity Outlier Report (MAD z-score)\n\n")
            f.write("Entries where |MAD z| ≥ 3.5\n\n")
            for r in outlier_records:
                f.write(f"- Prompt ID `{r['pid']}` | Paraphrase `{r['p_key']}` | MAD z = `{r['mad_z']:+.3f}`\n")

    # Dump component activation norms if captured
    if args.capture_component_activations:
        rows = []
        for pkey, compmap in seg_norm_base.items():
            for comp, layermap in compmap.items():
                for l, vals in layermap.items():
                    for v in vals:
                        rows.append({"paraphrase": pkey, "component": comp, "layer": l, "model": "base", "value": v})
        for pkey, compmap in seg_norm_ft.items():
            for comp, layermap in compmap.items():
                for l, vals in layermap.items():
                    for v in vals:
                        rows.append({"paraphrase": pkey, "component": comp, "layer": l, "model": "ft", "value": v})
        if rows:
            pd.DataFrame(rows).to_csv(output_path / "component_activation_norms_paraphrase_per_layer.csv", index=False)

        # Dump component activation cosines if captured
        rows = []
        for pkey, compmap in seg_cos_base.items():
            for comp, layermap in compmap.items():
                for l, vals in layermap.items():
                    for v in vals:
                        rows.append({"paraphrase": pkey, "component": comp, "layer": l, "model": "base", "value": v})
        for pkey, compmap in seg_cos_ft.items():
            for comp, layermap in compmap.items():
                for l, vals in layermap.items():
                    for v in vals:
                        rows.append({"paraphrase": pkey, "component": comp, "layer": l, "model": "ft", "value": v})
        if rows:
            pd.DataFrame(rows).to_csv(output_path / "component_activation_cosine_paraphrase_vs_original_per_layer.csv", index=False)

    # Aggregate plots + numbers
    num_processed = len(set(pid for pid, *_ in pairs))
    if num_processed > 0:
        plot_aggregate_results(aggregate_data, num_processed, output_path)

        # Aggregate numeric report
        all_dfs = {
            "Embedding Similarity": pd.DataFrame([(p, v, 'base') for p, V in aggregate_data['base_emb_sims'].items() for v in V] + 
                                                [(p, v, 'ft') for p, V in aggregate_data['ft_emb_sims'].items() for v in V], 
                                                columns=['paraphrase', 'value', 'model']),
            "Representational Similarity": pd.DataFrame([(p, l, v, 'base') for p, L in aggregate_data['base_sims'].items() for l, V in L.items() for v in V] +
                                                        [(p, l, v, 'ft') for p, L in aggregate_data['ft_sims'].items() for l, V in L.items() for v in V],
                                                        columns=['paraphrase', 'layer', 'value', 'model']),
            "Activation Norm": pd.DataFrame([(p, l, v, 'base') for p, L in aggregate_data['base_norms'].items() for l, V in L.items() for v in V] +
                                            [(p, l, v, 'ft') for p, L in aggregate_data['ft_norms'].items() for l, V in L.items() for v in V],
                                            columns=['paraphrase', 'layer', 'value', 'model']),
            "Pairwise L2 (orig↔para)": pd.DataFrame([(p, l, v, 'base') for p, L in aggregate_data['base_l2'].items() for l, V in L.items() for v in V] +
                                                    [(p, l, v, 'ft')   for p, L in aggregate_data['ft_l2'].items() for l, V in L.items() for v in V],
                                                    columns=['paraphrase', 'layer', 'value', 'model'])
        }
        report_md = ["# Aggregate Numerical Analysis Report", f"Based on **{num_processed}** processed prompts.", "---"]
        report_json = {"num_samples": num_processed, "analyses": {}}
        for analysis_name, df in all_dfs.items():
            if df.empty: continue
            report_md.append(f"## Analysis: `{analysis_name}`")
            report_md.append("| Paraphrase Type | Mean Base | Mean FT | Mean Δ | Median Δ | Std Δ | T-test p-value (Δ=0) | Significant? |")
            report_md.append("|---|---|---|---|---|---|---|---|")
            analysis_stats = {}
            all_base_vals, all_ft_vals = [], []
            p_keys = df['paraphrase'].unique()
            for p_key in sorted(p_keys):
                base_series = df[(df['paraphrase'] == p_key) & (df['model'] == 'base')]['value']
                ft_series = df[(df['paraphrase'] == p_key) & (df['model'] == 'ft')]['value']
                if len(base_series) < 2 or len(ft_series) < 2: continue
                all_base_vals.extend(base_series)
                all_ft_vals.extend(ft_series)
                delta_series = ft_series.values - base_series.values
                mean_base, mean_ft = base_series.mean(), ft_series.mean()
                mean_delta, median_delta, std_delta = delta_series.mean(), np.median(delta_series), delta_series.std()
                ttest_res = stats.ttest_1samp(delta_series, 0)
                is_sig = ttest_res.pvalue < 0.05
                report_md.append(f"| `{p_key}` | `{mean_base:.6f}` | `{mean_ft:.6f}` | `{mean_delta:+.6f}` | `{median_delta:+.6f}` | `{std_delta:.6f}` | `{ttest_res.pvalue:.6f}` | **{'YES' if is_sig else 'NO'}** |")
                analysis_stats[p_key] = {"mean_base": mean_base, "mean_ft": mean_ft, "mean_delta": mean_delta, "median_delta": median_delta, "std_dev_delta": std_delta, "p_value_delta": ttest_res.pvalue}
            if len(all_base_vals) >= 2:
                g_delta = np.array(all_ft_vals) - np.array(all_base_vals)
                g_mean_base, g_mean_ft = np.mean(all_base_vals), np.mean(all_ft_vals)
                g_mean_delta, g_median_delta, g_std_delta = g_delta.mean(), np.median(g_delta), g_delta.std()
                g_ttest = stats.ttest_1samp(g_delta, 0)
                g_is_sig = g_ttest.pvalue < 0.05
                report_md.append(f"| **_GRAND TOTAL_** | `{g_mean_base:.6f}` | `{g_mean_ft:.6f}` | `{g_mean_delta:+.6f}` | `{g_median_delta:+.6f}` | `{g_std_delta:.6f}` | `{g_ttest.pvalue:.6f}` | **{'YES' if g_is_sig else 'NO'}** |")
                analysis_stats["__grand_total__"] = {"mean_base": g_mean_base, "mean_ft": g_mean_ft, "mean_delta": g_mean_delta, "median_delta": g_median_delta, "std_dev_delta": g_std_delta, "p_value_delta": g_ttest.pvalue}
            report_md.append("\n---\n")
            report_json["analyses"][analysis_name] = analysis_stats
        (output_path / "aggregate_summary.md").write_text("\n".join(report_md), encoding="utf-8")
        (output_path / "aggregate_summary.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")
        script_logger.info("Saved aggregate numerical reports (MD and JSON).")

        # Correlations with ΔTF (optional)
        if per_item_rows_for_corr and args.scores_base_json_path and args.scores_ft_json_path:
            scores_base = load_scores(args.scores_base_json_path)
            scores_ft   = load_scores(args.scores_ft_json_path)
            compute_correlations(per_item_rows_for_corr, scores_base, scores_ft, output_path)

    # Best-vs-Worst experiment
    if args.scores_base_json_path:
        scores_base = load_scores(args.scores_base_json_path)
        cache2 = cache or AnswerCache()
        scores_ft = load_scores(args.scores_ft_json_path) if args.scores_ft_json_path else {}
        run_best_worst_experiment(
            base_model, ft_model, tokenizer, prompts_data, scores_base, scores_ft, device, output_path,
            gen_kwargs, cache2, base_model_id, ft_model_id
        )

# Sanity checks
@torch.no_grad()
def run_sanity_checks(base_model: PreTrainedModel, tokenizer: AutoTokenizer,
                      prompts_data: List[Dict[str, Any]], device: str, output_path: Path,
                      focus_on_answer: bool, gen_kwargs: dict, cache: Optional[AnswerCache], base_model_id: str):
    """
    Base vs Base self-comparison: orig vs orig, para vs para. Should be ~0 distance.
    We'll test a small subset for speed.
    """
    rows = []
    max_samples = 25  # small sanity subset
    count = 0
    for item in prompts_data:
        if count >= max_samples: break
        pid = item.get("prompt_count"); inp = item.get("input", "")
        orig = item.get("instruction_original")
        if not orig: continue
        # pick first paraphrase available
        parakeys = [k for k in item if k.startswith("instruct_") and item[k]]
        if not parakeys: continue
        para = item[parakeys[0]]
        ans_o = ans_p = None
        if focus_on_answer:
            ans_o = generate_answer_text(base_model, tokenizer, orig, inp, device, cache=cache, model_id=base_model_id, **gen_kwargs)
            ans_p = generate_answer_text(base_model, tokenizer, para, inp, device, cache=cache, model_id=base_model_id, **gen_kwargs)

        # Encode orig vs orig
        enc1a, enc1b = encode_pair(tokenizer, orig, orig, inp, device, ans_o, ans_o)
        # Encode para vs para
        enc2a, enc2b = encode_pair(tokenizer, para, para, inp, device, ans_p, ans_p)

        batch = collate_batch([enc1a, enc1b, enc2a, enc2b], device)
        with ActivationExtractor(base_model) as ex:
            _ = base_model(batch.input_ids, attention_mask=batch.attention_mask,
                   use_cache=True)
            hs = dict(sorted(ex.hidden_states.items()))

        mask = batch.seg_answer if focus_on_answer and batch.seg_answer.any() else batch.attention_mask

        # Distances
        l2_o = l2_by_layer_masked(hs, 0, 1, mask, mask)  # orig vs orig
        l2_p = l2_by_layer_masked(hs, 2, 3, mask, mask)  # para vs para
        final_layer = max(l2_o.keys())
        rows.append({"prompt_count": pid, "l2_orig_orig_final": l2_o[final_layer], "l2_para_para_final": l2_p[final_layer]})
        count += 1

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path / "sanity_base_self_similarity.csv", index=False)

def list_param_names(model: PreTrainedModel, contains: str):
    for n, _ in model.named_parameters():
        if contains in n:
            print(n)

@torch.no_grad()
def inspect_param_stats(base_model: PreTrainedModel, ft_model: PreTrainedModel, name: str):
    bp = dict(base_model.named_parameters()).get(name, None)
    fp = dict(ft_model.named_parameters()).get(name, None)
    if bp is None or fp is None:
        print(f"Param not found in both models: {name}")
        return
    b = bp.detach().float().view(-1)
    f = fp.detach().float().view(-1)
    d = (f - b)
    base_fro = torch.linalg.vector_norm(b).item()
    ft_fro   = torch.linalg.vector_norm(f).item()
    delta_fro= torch.linalg.vector_norm(d).item()
    mean_abs = d.abs().mean().item()
    print(f"[{name}]")
    print(f"  ||W_base||_F = {base_fro:.6f}")
    print(f"  ||W_ft||_F   = {ft_fro:.6f}")
    print(f"  ||ΔW||_F     = {delta_fro:.6f}")
    print(f"  mean|Δ|      = {mean_abs:.6e}")


# Main
def main():
    parser = argparse.ArgumentParser(description="Robustness Analyzer (batched, component-aware, FA2 optional)", formatter_class=argparse.RawTextHelpFormatter)
    # Paths
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--ft_model_path", type=str, required=True)
    parser.add_argument("--prompts_json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="robustness_analysis_results")
    # Modes
    parser.add_argument("--run_mode", type=str, choices=['aggregate', 'sanity_only'], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--paraphrase_keys", type=str, nargs='+')
    parser.add_argument("--data_split", type=str, choices=['all', 'train', 'val', 'test'], default='test')
    parser.add_argument("--val_size", type=float, default=0.05)
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    # Analysis options
    parser.add_argument("--compute_pii", action="store_true")
    parser.add_argument("--pii_focus_on_answer", action="store_true")
    parser.add_argument("--compute_svd", action="store_true")
    parser.add_argument("--topk_weights", type=int, default=10)
    parser.add_argument("--capture_component_activations", action="store_true", help="Capture attn.o_proj and mlp.down_proj outputs and record norms.")
    # Scoring (optional)
    parser.add_argument("--scores_base_json_path", type=str, default="")
    parser.add_argument("--scores_ft_json_path", type=str, default="")
    # Generation / Answer focus
    parser.add_argument("--focus_on_answer_tokens", action="store_true", help="Generate answers and restrict analyses to produced answer tokens.")
    parser.add_argument("--gen_max_new_tokens", type=int, default=128)
    parser.add_argument("--gen_temperature", type=float, default=0.0)
    parser.add_argument("--gen_top_p", type=float, default=1.0)
    # Batching & perf
    parser.add_argument("--pairs_per_batch", type=int, default=16, help="Number of (orig,para) PAIRS processed per forward (per model). Effective batch size is 2x this.")
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--flash_attention", type=str, choices=["auto", "flash_attention_2", "sdpa", "off"], default="auto")
    parser.add_argument("--capture_attn_qkv", action="store_true",
                        help="Also capture Q/K/V projection outputs")
    parser.add_argument("--capture_mlp_up_gate", action="store_true",
                        help="Also capture MLP up_proj and gate_proj outputs")

    parser.add_argument("--merge_lora_for_weights", action="store_true",
                        help="If the FT path is a PEFT/LoRA adapter, merge adapters into base weights for weight analysis.")

    # DON'T! takes too long
    parser.add_argument("--cosine_upcast_fp32", action="store_true",
                        help="If set, upcast tensors to float32 before cosine similarity. Default: keep original dtype (e.g., bf16).")

    parser.add_argument("--list_params_containing", type=str, default="",
                        help="Print all parameter names containing this substring and exit.")
    parser.add_argument("--inspect_param", type=str, default="",
                        help="Print norms for an exact parameter name and exit.")

    parser.add_argument("--skip_weights_delta_per_layer_plots", action="store_true",
                        help="Skip generating weights_delta_fro_per_layer__*.png plots.")
    parser.add_argument("--skip_weights_one_minus_cosine_per_layer_plots", action="store_true",
                        help="Skip generating weights_one_minus_cosine_per_layer__*.png plots (including overall).")
    parser.add_argument("--skip_weights_rel_delta_per_layer_plots", action="store_true",
                        help="Skip generating weights_rel_delta_per_layer__*.png plots.")

    args = parser.parse_args()

    global UPCAST_FP32
    UPCAST_FP32 = args.cosine_upcast_fp32

    output_path = Path(args.output_dir); output_path.mkdir(exist_ok=True, parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        script_logger.warning("CUDA not available. Running on CPU will be extremely slow.")

    attn_impl = enable_perf_features(args)

    script_logger.info("--- Model and Data Loading ---")
    base_model, tokenizer, base_model_id = load_model_and_tokenizer(
        args.base_model_path, args.base_model_path, device, attn_impl, dtype_str=args.dtype,
        merge_lora_for_weights=args.merge_lora_for_weights
    )
    ft_model, _tok2, ft_model_id = load_model_and_tokenizer(
        args.ft_model_path, args.base_model_path, device, attn_impl, dtype_str=args.dtype,
        merge_lora_for_weights=args.merge_lora_for_weights
    )

    if args.list_params_containing or args.inspect_param:
        if args.list_params_containing:
            print(f"Parameters containing '{args.list_params_containing}':")
            list_param_names(ft_model, args.list_params_containing)
        if args.inspect_param:
            inspect_param_stats(base_model, ft_model, args.inspect_param)
        return

    with open(args.prompts_json_path, 'r', encoding='utf-8') as f:
        prompts_data_all = json.load(f)

    if args.data_split != 'all':
        all_pids = [item['prompt_count'] for item in prompts_data_all if 'prompt_count' in item]
        train_ids, val_ids, test_ids = get_group_wise_split_ids(all_pids, args.val_size, args.test_size, args.seed)
        target_ids = {'train': train_ids, 'val': val_ids, 'test': test_ids}[args.data_split]
        prompts_data = [item for item in prompts_data_all if item.get('prompt_count') in target_ids]
    else:
        prompts_data = prompts_data_all

    if args.limit > 0:
        prompts_data = prompts_data[:args.limit]
        script_logger.info(f"Limited run to the first {len(prompts_data)} items.")

    # Sanity-only mode
    cache = AnswerCache() if (args.focus_on_answer_tokens or args.pii_focus_on_answer) else None
    gen_kwargs = dict(max_new_tokens=args.gen_max_new_tokens, temperature=args.gen_temperature, top_p=args.gen_top_p)

    if args.run_mode == 'sanity_only':
        run_sanity_checks(base_model, tokenizer, prompts_data, device, output_path,
                          focus_on_answer=args.focus_on_answer_tokens, gen_kwargs=gen_kwargs, cache=cache, base_model_id=base_model_id)
        script_logger.info(f"Sanity checks complete. Outputs at: {output_path}")
        return

    # Aggregate mode
    run_aggregate(base_model, ft_model, tokenizer, prompts_data, device, output_path, args, base_model_id, ft_model_id)
    script_logger.info(f"Analysis complete. All outputs saved to: {output_path}")

if __name__ == "__main__":
    main()


