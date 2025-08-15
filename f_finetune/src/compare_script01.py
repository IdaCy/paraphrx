#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extended Script 01: Activation/Embedding/Weight Diff Analyzer (masked + segmented + PII + module ΔW + correlations)

Keeps all original outputs, adds:
- Masked pooling (no pad leakage)
- Segment-wise metrics (instruction / input / assistant_prefix if chat template supports)
- Paraphrase Invariance Index (PII) per layer (normalized by unrelated pair)
- Detailed weight delta breakdown by module + per-parameter normalization + optional SVD summaries
- Robust outlier detection (MAD z-scores)
- Optional correlations of Δmetrics with ΔTF via scores JSONs


  python 01_activation_similarity_norms.py \
    --base_model_path <BASE> \
    --ft_model_path <FT> \
    --prompts_json_path <PROMPTS> \
    --run_mode aggregate \
    --output_dir results \
    [--scores_base_json_path <SCORES_BASE.json>] \
    [--scores_ft_json_path <SCORES_FT.json>] \
    [--compute_pii] [--compute_svd] [--paraphrase_keys instruct_* ...]

python3 f_finetune/src/compare_script01.py \
    --base_model_path f_finetune/model \
    --ft_model_path f_finetune/outputs_great_nolap/lpr9x_a1_notarg_50k_ft/final \
    --prompts_json_path a_data/alpaca/50k_phrxed.json \
    --output_dir f_finetune/outputs_great_nolap/lpr9x_a1_notarg_50k_ft/masstest100 \
    --run_mode aggregate \
    --limit 100 \
    --compute_pii --compute_svd \
    --scores_base_json_path c_assess_inf/output/alpaca_answer_scores/gemma-2-2b-it.json \
    --scores_ft_json_path f_finetune/outputs_great_nolap/lpr9x_a1_notarg_50k_ft/scores.json

    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--ft_model_path", type=str, required=True)
    parser.add_argument("--prompts_json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="robustness_analysis_results")
    parser.add_argument("--run_mode", type=str, choices=['case_study', 'aggregate'], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--paraphrase_keys", type=str, nargs='+')
    parser.add_argument("--prompt_ids", type=int, nargs='+')
    parser.add_argument("--data_split", type=str, choices=['all', 'train', 'val', 'test'], default='test')
    parser.add_argument("--val_size", type=float, default=0.05)
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    # options
    parser.add_argument("--compute_pii", action="store_true", help="Compute Paraphrase Invariance Index per layer (Base & FT).")
    parser.add_argument("--compute_svd", action="store_true", help="Compute SVD summaries for 2D weight deltas.")
    parser.add_argument("--scores_base_json_path", type=str, default="", help="Scores JSON for base (TF at index 0).")
    parser.add_argument("--scores_ft_json_path", type=str, default="", help="Scores JSON for FT (TF at index 0).")
"""

import argparse
import json
import logging
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from peft import PeftModel
from scipy import stats
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedModel)

# Logging
class SafeFormatter(logging.Formatter):
    def format(self, record):
        record_copy = logging.makeLogRecord(record.__dict__)
        try:
            return super().format(record_copy)
        except (TypeError, ValueError):
            record_copy.msg = f"MALFORMED LOG: {record_copy.msg} | ARGS: {record_copy.args}"
            record_copy.args = ()
            return logging.Formatter('[%(levelname)s] %(message)s').format(record_copy)

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

# Plot Style
plt.switch_backend('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

# Helpers
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: [seq, hidden] or [batch, seq, hidden]
    mask: [seq] or [batch, seq] with 1 for valid tokens
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

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.view(1, -1), b.view(1, -1)).item()

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

# Prompt / Segments
def build_user_message(instruction: str, inp: str) -> str:
    return instruction if not inp else f"{instruction}\n\nInput:\n{inp}"

def tokenise_with_segments(tokenizer: AutoTokenizer, instruction: str, inp: str, device: str):
    """
    Returns dict with input_ids, attention_mask, and boolean masks for segments:
      - seg_instruction
      - seg_input (may be zeros if empty)
      - seg_asst_prefix (assistant BOS / prefix if chat template supports and we add it)
    We attempt to use chat template (Gemma-2-IT style). If not available, we fall back to a simple "### Instruction"/"### Input" format.
    """
    # Try chat template first
    user_msg = build_user_message(instruction, inp)
    segs = {}
    try:
        messages = [{"role": "user", "content": user_msg}]
        # Without assistant prefix
        plain_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")
        # With assistant prefix (so last tokens are assistant prefix)
        asst_ids  = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        # Compute prefix lengths to define assistant segment length
        seg_len_plain = plain_ids.shape[1]
        seg_len_asst  = asst_ids.shape[1]
        asst_prefix_len = seg_len_asst - seg_len_plain
        input_ids = asst_ids.to(device)
        attention_mask = torch.ones_like(input_ids)
        # approximate instruction/input split by tokenising parts separately
        instr_only = tokenizer.apply_chat_template([{"role":"user","content": instruction}], tokenize=True, add_generation_prompt=False, return_tensors="pt")
        len_instr_only = instr_only.shape[1]
        # compute an approximate input segment len by difference when input is added
        if inp:
            user_w_input = tokenizer.apply_chat_template([{"role":"user","content": user_msg}], tokenize=True, add_generation_prompt=False, return_tensors="pt")
            len_user_w_input = user_w_input.shape[1]
            input_len = max(0, len_user_w_input - len_instr_only)
        else:
            input_len = 0
        # boundaries (all within plain length)
        seg_instruction = torch.zeros(seg_len_asst, dtype=torch.bool)
        seg_input = torch.zeros(seg_len_asst, dtype=torch.bool)
        seg_asst_prefix = torch.zeros(seg_len_asst, dtype=torch.bool)

        seg_instruction[:min(len_instr_only, seg_len_asst)] = True
        if input_len > 0:
            start = len_instr_only
            end = min(start + input_len, seg_len_asst)
            seg_input[start:end] = True
        if asst_prefix_len > 0:
            seg_asst_prefix[-asst_prefix_len:] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask.to(device),
            "seg_instruction": seg_instruction.unsqueeze(0).to(device),
            "seg_input": seg_input.unsqueeze(0).to(device),
            "seg_asst_prefix": seg_asst_prefix.unsqueeze(0).to(device),
        }
    except Exception:
        # Fallback to "###" format with explicit boundaries
        prefix_instr = "### Instruction:\n"
        prefix_input = "### Input:\n"
        text = prefix_instr + instruction + "\n\n"
        instr_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if inp:
            text2 = prefix_input + inp + "\n\n"
            input_ids_extra = tokenizer(text2, add_special_tokens=False, return_tensors="pt")["input_ids"]
            input_ids = torch.cat([instr_ids, input_ids_extra], dim=1)
            seg_instruction = torch.zeros(input_ids.shape[1], dtype=torch.bool); seg_instruction[:, :instr_ids.shape[1]] = True
            seg_input = torch.zeros(input_ids.shape[1], dtype=torch.bool); seg_input[:, instr_ids.shape[1]:] = True
        else:
            input_ids = instr_ids
            seg_instruction = torch.ones(input_ids.shape[1], dtype=torch.bool)
            seg_input = torch.zeros(input_ids.shape[1], dtype=torch.bool)
        # no assistant prefix in fallback
        seg_asst_prefix = torch.zeros(input_ids.shape[1], dtype=torch.bool)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "seg_instruction": seg_instruction.unsqueeze(0).to(device),
            "seg_input": seg_input.unsqueeze(0).to(device),
            "seg_asst_prefix": seg_asst_prefix.unsqueeze(0).to(device),
        }

# Weight ΔW (detailed)
def module_bucket(name: str) -> Tuple[str, str, int]:
    """
    Return (layer_bucket, module_bucket, layer_index) for parameter name.
    """
    layer_idx = -1
    if 'layers.' in name:
        try:
            layer_idx = int(name.split('layers.')[1].split('.')[0])
        except Exception:
            layer_idx = -1
    # buckets
    if any(k in name for k in ['q_proj','k_proj','v_proj','o_proj']):
        return ('layer', 'attn_qkvo', layer_idx)
    if any(k in name for k in ['up_proj','down_proj','gate_proj']):
        return ('layer', 'mlp_up_down_gate', layer_idx)
    if 'norm' in name:
        return ('layer', 'norms', layer_idx)
    if 'embed' in name or 'tok_embeddings' in name:
        return ('global', 'embeddings', -1)
    if 'lm_head' in name:
        return ('global', 'lm_head', -1)
    return ('other', 'other', layer_idx)

def analyze_and_plot_weight_deltas(base_model: PreTrainedModel, ft_model: PreTrainedModel, output_path: Path,
                                   compute_svd: bool = False, topk: int = 5):
    script_logger.info("Comparing model weights layer by layer (detailed)...")
    base_params = dict(base_model.named_parameters())
    ft_params = dict(ft_model.named_parameters())

    rows = []
    per_layer_sum = defaultdict(float)

    for name, ft_param in ft_params.items():
        if name not in base_params: continue
        b = base_params[name]
        if ft_param.shape != b.shape: continue
        with torch.no_grad():
            diff = (ft_param.to(b.device) - b).detach()
            delta_frob = torch.linalg.norm(diff).item()
            n_params = diff.numel()
            mean_abs = diff.abs().mean().item()
            per_layer_key = None
            if 'layers.' in name:
                try:
                    lnum = int(name.split('layers.')[1].split('.')[0])
                    per_layer_key = lnum
                    per_layer_sum[lnum] += delta_frob
                except Exception:
                    pass
            # SVD (optional, only for 2D matrices)
            svd_vals = []
            energy_ratio_topk = None
            # Guard against large matrices and unsupported dtypes for stability and performance.
            SIZE_LIMIT = 16384  # Skip SVD for matrices with any dimension larger than this.
            if compute_svd and diff.dim() == 2 and min(diff.shape) >= topk and max(diff.shape) <= SIZE_LIMIT:
                try:
                    # 1. Cast to float32 on CPU for SVD compatibility, as bfloat16 is not supported.
                    M = diff.to(dtype=torch.float32, device='cpu')
                    # 2. Compute SVD on the correctly typed matrix.
                    sv = torch.linalg.svdvals(M)
                    sv = sv.detach().cpu().numpy()
                    sv_sorted = np.sort(sv)[::-1]
                    top = sv_sorted[:topk]
                    svd_vals = top.tolist()
                    # Add a small epsilon to prevent division by zero for zero-matrices.
                    energy_ratio_topk = float((top**2).sum() / ((sv_sorted**2).sum() + 1e-9))
                except Exception as e:
                    script_logger.warning(f"SVD failed for {name} (shape: {list(diff.shape)}): {e}")

            scope, bucket, layer_idx = module_bucket(name)
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
                "svd_topk": svd_vals,
                "topk_energy_ratio": energy_ratio_topk
            })

    df = pd.DataFrame(rows)
    if df.empty:
        script_logger.warning("No comparable parameters for ΔW analysis.")
        return

    # Save CSVs
    df.to_csv(output_path / "weight_deltas_detailed.csv", index=False)

    # Aggregates
    agg_layer = df[df["layer_idx"] >= 0].groupby("layer_idx")["delta_fro"].sum().reset_index()
    agg_bucket = df.groupby(["bucket"])["delta_fro"].sum().reset_index().sort_values("delta_fro", ascending=False)
    agg_bucket_normed = df.groupby(["bucket"])["delta_per_param"].mean().reset_index().sort_values("delta_per_param", ascending=False)

    agg_layer.to_csv(output_path / "weight_deltas_per_layer.csv", index=False)
    agg_bucket.to_csv(output_path / "weight_deltas_per_bucket_sum.csv", index=False)
    agg_bucket_normed.to_csv(output_path / "weight_deltas_per_bucket_normalized.csv", index=False)

    # Legacy per-layer bar
    if not agg_layer.empty:
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.bar(agg_layer["layer_idx"], agg_layer["delta_fro"], color='indigo', alpha=0.8)
        ax.set_title('Fine-Tuning Impact: L2 Norm of Weight Deltas per Layer', fontsize=16, weight='bold')
        ax.set_xlabel('Decoder Layer'); ax.set_ylabel('Sum of L2 Norms of Parameter Deltas')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        fig.savefig(output_path / "analysis_weight_deltas.png")
        plt.close(fig)

    # bucket plot
    if not agg_bucket.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(agg_bucket["bucket"], agg_bucket["delta_fro"], alpha=0.85)
        ax.set_title("ΔW by Module Bucket (sum Frobenius)", fontsize=14, weight='bold')
        ax.set_xlabel("Σ ||ΔW||_F"); ax.set_ylabel("Module")
        plt.tight_layout()
        fig.savefig(output_path / "analysis_weight_deltas_by_bucket.png")
        plt.close(fig)

# Activation Capture
class ActivationExtractor:
    def __init__(self, model: PreTrainedModel):
        self._model = model
        self.hidden_states = {}
        self._hook_handles = []
    def _hook_fn(self, layer_idx: int):
        def hook(module, input, output):
            # output[0]: hidden states [batch, seq, hidden]
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
        for handle in self._hook_handles: handle.remove()
        self._hook_handles.clear()

# Pair Analysis (masked + segmented + PII hooks)
@torch.no_grad()
def forward_collect(model: PreTrainedModel, tokenizer: AutoTokenizer, instruction: str, paraphrase: str, inp: str, device: str):
    # Build segmented inputs for orig and paraphrase
    enc_o = tokenise_with_segments(tokenizer, instruction, inp, device)
    enc_p = tokenise_with_segments(tokenizer, paraphrase,  inp, device)

    # embeddings (masked mean)
    emb = model.get_input_embeddings()
    emb_o = emb(enc_o["input_ids"])
    emb_p = emb(enc_p["input_ids"])

    emb_o_mean_all = masked_mean(emb_o[0], enc_o["attention_mask"][0])
    emb_p_mean_all = masked_mean(emb_p[0], enc_p["attention_mask"][0])

    # segment means
    def seg_means(emb_x, enc):
        m = {}
        for seg_name in ["seg_instruction", "seg_input", "seg_asst_prefix"]:
            seg_mask = (enc[seg_name] & enc["attention_mask"].bool()).squeeze(0)
            if seg_mask.any():
                m[seg_name] = masked_mean(emb_x[0], seg_mask)
            else:
                m[seg_name] = None
        return m

    emb_o_segs = seg_means(emb_o, enc_o)
    emb_p_segs = seg_means(emb_p, enc_p)

    # hidden states
    with ActivationExtractor(model) as ex:
        _ = model(enc_o["input_ids"], attention_mask=enc_o["attention_mask"])
        hs_o = dict(sorted(ex.hidden_states.items()))
    with ActivationExtractor(model) as ex:
        _ = model(enc_p["input_ids"], attention_mask=enc_p["attention_mask"])
        hs_p = dict(sorted(ex.hidden_states.items()))

    return {
        "enc_o": enc_o, "enc_p": enc_p,
        "emb_o_all": emb_o_mean_all, "emb_p_all": emb_p_mean_all,
        "emb_o_segs": emb_o_segs, "emb_p_segs": emb_p_segs,
        "hs_o": hs_o, "hs_p": hs_p
    }

def cosine_by_layer_masked(hs_o: Dict[int, torch.Tensor], hs_p: Dict[int, torch.Tensor],
                           mask_o: torch.Tensor, mask_p: torch.Tensor) -> Dict[int, float]:
    sims = {}
    # masked mean per layer, then cosine
    for l in hs_o.keys():
        h_o = hs_o[l][0]  # [seq, hidden]
        h_p = hs_p[l][0]
        mo = mask_o.squeeze(0).bool()
        mp = mask_p.squeeze(0).bool()
        h_o_mean = masked_mean(h_o, mo)
        h_p_mean = masked_mean(h_p, mp)
        sims[l] = cosine(h_o_mean, h_p_mean)
    return sims

def norms_by_layer_masked(hs: Dict[int, torch.Tensor], mask: torch.Tensor) -> Dict[int, float]:
    norms = {}
    m = mask.squeeze(0).bool()
    for l, h in hs.items():
        v = masked_mean(h[0], m)  # [hidden]
        norms[l] = torch.linalg.norm(v.float()).item()
    return norms

@torch.no_grad()
def run_and_analyze_pair(base_model: PreTrainedModel, ft_model: PreTrainedModel,
                         tokenizer: AutoTokenizer, original_text: str, paraphrase_text: str,
                         inp: str, device: str) -> Dict[str, Any]:

    base = forward_collect(base_model, tokenizer, original_text, paraphrase_text, inp, device)
    ft   = forward_collect(ft_model,   tokenizer, original_text, paraphrase_text, inp, device)

    # Embedding similarities (all + segments)
    base_emb_sim_all = cosine(base["emb_o_all"], base["emb_p_all"])
    ft_emb_sim_all   = cosine(ft["emb_o_all"],   ft["emb_p_all"])

    seg_sims_base, seg_sims_ft = {}, {}
    for seg in ["seg_instruction","seg_input","seg_asst_prefix"]:
        eo, ep = base["emb_o_segs"][seg], base["emb_p_segs"][seg]
        fo, fp = ft["emb_o_segs"][seg],   ft["emb_p_segs"][seg]
        if eo is not None and ep is not None:
            seg_sims_base[seg] = cosine(eo, ep)
        if fo is not None and fp is not None:
            seg_sims_ft[seg]   = cosine(fo, fp)

    # Layerwise similarities (all + segments)
    base_cos_all = cosine_by_layer_masked(base["hs_o"], base["hs_p"],
                                          base["enc_o"]["attention_mask"], base["enc_p"]["attention_mask"])
    ft_cos_all   = cosine_by_layer_masked(ft["hs_o"], ft["hs_p"],
                                          ft["enc_o"]["attention_mask"], ft["enc_p"]["attention_mask"])

    # Norms (masked means)
    base_norms_para_all = norms_by_layer_masked(base["hs_p"], base["enc_p"]["attention_mask"])
    ft_norms_para_all   = norms_by_layer_masked(ft["hs_p"],   ft["enc_p"]["attention_mask"])

    # Segment-wise norms on paraphrase
    base_norms_para_segs, ft_norms_para_segs = {}, {}
    for seg in ["seg_instruction","seg_input","seg_asst_prefix"]:
        seg_mask_b = base["enc_p"][seg] & base["enc_p"]["attention_mask"].bool()
        seg_mask_f = ft["enc_p"][seg]   & ft["enc_p"]["attention_mask"].bool()
        # compute only if any token in segment
        if seg_mask_b.squeeze(0).any():
            base_norms_para_segs[seg] = norms_by_layer_masked(base["hs_p"], seg_mask_b)
        if seg_mask_f.squeeze(0).any():
            ft_norms_para_segs[seg]   = norms_by_layer_masked(ft["hs_p"], seg_mask_f)

    return {
        "base_emb_sim": base_emb_sim_all,
        "ft_emb_sim": ft_emb_sim_all,
        "base_emb_seg_sims": seg_sims_base,
        "ft_emb_seg_sims": seg_sims_ft,
        "base_cos_sims": base_cos_all,
        "ft_cos_sims": ft_cos_all,
        "base_norms_para": base_norms_para_all,
        "ft_norms_para": ft_norms_para_all,
        "base_norms_para_segs": base_norms_para_segs,
        "ft_norms_para_segs": ft_norms_para_segs
    }

# PII (Paraphrase Invariance Index)
@torch.no_grad()
def compute_pii_for_item(model: PreTrainedModel, tokenizer: AutoTokenizer,
                         instruction: str, paraphrase: str, unrelated_instr: str,
                         inp: str, device: str) -> Dict[int, float]:
    """
    PII_l = 1 - (cos(orig, paraphrase) / cos(orig, unrelated)), computed from masked-mean hidden states per layer.
    Higher PII (after FT) => greater paraphrase invariance relative to unrelated baseline.
    """
    enc_o = tokenise_with_segments(tokenizer, instruction, inp, device)
    enc_p = tokenise_with_segments(tokenizer, paraphrase,  inp, device)
    enc_u = tokenise_with_segments(tokenizer, unrelated_instr, inp, device)

    with ActivationExtractor(model) as ex:
        _ = model(enc_o["input_ids"], attention_mask=enc_o["attention_mask"])
        h_o = dict(sorted(ex.hidden_states.items()))
    with ActivationExtractor(model) as ex:
        _ = model(enc_p["input_ids"], attention_mask=enc_p["attention_mask"])
        h_p = dict(sorted(ex.hidden_states.items()))
    with ActivationExtractor(model) as ex:
        _ = model(enc_u["input_ids"], attention_mask=enc_u["attention_mask"])
        h_u = dict(sorted(ex.hidden_states.items()))

    pii = {}
    mo = enc_o["attention_mask"]; mp = enc_p["attention_mask"]; mu = enc_u["attention_mask"]
    for l in h_o.keys():
        o = masked_mean(h_o[l][0], mo[0].bool())
        p = masked_mean(h_p[l][0], mp[0].bool())
        u = masked_mean(h_u[l][0], mu[0].bool())
        cos_op = torch.nn.functional.cosine_similarity(o.view(1,-1), p.view(1,-1)).item()
        cos_ou = torch.nn.functional.cosine_similarity(o.view(1,-1), u.view(1,-1)).item()
        # Stabilize small denominators
        value = 1.0 - (cos_op / max(1e-6, cos_ou))
        pii[l] = value
    return pii

# Model Loading
def load_model_and_tokenizer(model_path: str, base_model_path: str, device: str) -> tuple[PreTrainedModel, AutoTokenizer]:
    path, base_path = Path(model_path), Path(base_model_path)
    if not base_path.exists(): script_logger.error(f"Base model path not found: {base_path}"); sys.exit(1)
    script_logger.info(f"Loading tokenizer from: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if (path / "adapter_config.json").exists():
        script_logger.info("Detected LoRA adapters. Loading base model and applying adapters.")
        model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map={"": device})
        model = PeftModel.from_pretrained(model, model_path)
    else:
        script_logger.info("Detected a full model. Loading directly.")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map={"": device})
    if model is None: script_logger.error(f"Could not load model from {model_path}"); sys.exit(1)
    model.eval()
    return model, tokenizer

def get_group_wise_split_ids(all_prompt_ids: List[int], val_size: float, test_size: float, seed: int) -> Tuple[Set[int], Set[int], Set[int]]:
    prompt_ids = sorted(list(set(all_prompt_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(prompt_ids)
    n_test, n_val = int(len(prompt_ids) * test_size), int(len(prompt_ids) * val_size)
    test_ids, val_ids = set(prompt_ids[:n_test]), set(prompt_ids[n_test : n_test + n_val])
    train_ids = set(prompt_ids[n_test + n_val:])
    script_logger.info(f"Group-wise split complete. Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    return train_ids, val_ids, test_ids

# Visualisation
def plot_case_study(results: Dict, prompt_id: int, p_key: str, output_path: Path):
    layers = list(results['base_cos_sims'].keys())
    base_sims, ft_sims = np.array(list(results['base_cos_sims'].values())), np.array(list(results['ft_cos_sims'].values()))
    sim_delta = ft_sims - base_sims
    base_norms, ft_norms = np.array(list(results['base_norms_para'].values())), np.array(list(results['ft_norms_para'].values()))
    norm_delta = ft_norms - base_norms

    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Comprehensive Case Study Analysis\nPrompt ID: {prompt_id} | Paraphrase: "{p_key}"', fontsize=20, weight='bold')
    axs[0, 0].plot(layers, base_sims, 'o-', label='Base Model Similarity', color='cornflowerblue', lw=2)
    axs[0, 0].plot(layers, ft_sims, 's-', label='Fine-Tuned Model Similarity', color='firebrick', lw=2)
    axs[0, 0].fill_between(layers, base_sims, ft_sims, where=(ft_sims > base_sims), color='mediumseagreen', alpha=0.3, label='Improvement')
    axs[0, 0].set_title('1. Representational Similarity Trajectory', fontsize=14); axs[0, 0].set_ylabel('Cosine Similarity'); axs[0, 0].legend()
    colors_sim = ['mediumseagreen' if d >= 0 else 'salmon' for d in sim_delta]
    axs[0, 1].bar(layers, sim_delta, color=colors_sim, alpha=0.8); axs[0, 1].axhline(0, color='black', lw=1, linestyle='--')
    axs[0, 1].set_title('2. Change in Similarity (Δ)', fontsize=14); axs[0, 1].set_ylabel('Δ Cosine Similarity (FT - Base)')
    axs[1, 0].plot(layers, base_norms, 'o-', label='Base Model Norms', color='darkorange', lw=2)
    axs[1, 0].plot(layers, ft_norms, 's-', label='Fine-Tuned Model Norms', color='purple', lw=2)
    axs[1, 0].set_title('3. Mean Activation Norms (Paraphrase Input)', fontsize=14); axs[1, 0].set_ylabel('Mean L2 Norm'); axs[1, 0].legend()
    colors_norm = ['mediumseagreen' if d >= 0 else 'salmon' for d in norm_delta]
    axs[1, 1].bar(layers, norm_delta, color=colors_norm, alpha=0.8); axs[1, 1].axhline(0, color='black', lw=1, linestyle='--')
    axs[1, 1].set_title('4. Change in Activation Norm (Δ)', fontsize=14); axs[1, 1].set_ylabel('Δ Mean L2 Norm (FT - Base)')
    for ax in axs.flat: ax.grid(True, which='both', linestyle='--', linewidth=0.5); ax.set_xlabel('Decoder Layer')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]); fig.savefig(output_path / f"case_study_prompt_{prompt_id}_{p_key}.png"); plt.close(fig)

# Aggregation Plots
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
        ax.fill_between(mean_by_layer.index, mean_by_layer.values - 1.96 * sem_by_layer, mean_by_layer.values + 1.96 * sem_by_layer, color='mediumseagreen', alpha=0.2, label='95% Confidence Interval')
        ax.axhline(0, color='black', lw=1, linestyle='--'); ax.legend(); ax.grid(True)
        ax.set_title(f'Overall Average Change in Representational Similarity (N={num_samples})', fontsize=16, weight='bold')
        ax.set_xlabel('Decoder Layer'); ax.set_ylabel('Average Δ Cosine Similarity (FT - Base)')
        plt.tight_layout(); fig.savefig(output_path / "aggregate_lineplot_sim_delta.png"); plt.close(fig)

    if not df_sim_delta.empty:
        heatmap_data = df_sim_delta.groupby(['paraphrase', 'layer'])['delta'].mean().unstack(level='layer')
        fig, ax = plt.subplots(figsize=(16, max(8, len(heatmap_data.index) * 0.5)))
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", ax=ax, cbar_kws={'label': 'Average Δ Cosine Similarity (FT - Base)'})
        ax.set_title(f'Aggregate Analysis: Mean Change in Similarity by Paraphrase (N={num_samples})', fontsize=16, weight='bold')
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
        sns.boxplot(data=df_emb_delta, x='delta', y='paraphrase', orient='h', ax=ax, palette='coolwarm')
        ax.axvline(0, color='black', lw=1, linestyle='--')
        ax.set_title(f'Change in Embedding Space Similarity After Fine-Tuning (N={num_samples})', fontsize=16, weight='bold')
        ax.set_xlabel('Δ Cosine Similarity (FT - Base)'); ax.set_ylabel('Paraphrase Type')
        ax.grid(True, axis='x'); plt.tight_layout(); fig.savefig(output_path / "aggregate_boxplot_embedding_delta.png"); plt.close(fig)
    script_logger.info("Saved all aggregate plots.")

# Reporting
def report_case_study_numerics(results: Dict, prompt_id: int, p_key: str, output_path: Path):
    base_sims, ft_sims = np.array(list(results['base_cos_sims'].values())), np.array(list(results['ft_cos_sims'].values()))
    sim_delta, sim_ttest = ft_sims - base_sims, stats.ttest_rel(ft_sims, base_sims)
    base_norms, ft_norms = np.array(list(results['base_norms_para'].values())), np.array(list(results['ft_norms_para'].values()))
    norm_delta, norm_ttest = ft_norms - base_norms, stats.ttest_rel(ft_norms, base_norms)
    emb_sim_delta = results['ft_emb_sim'] - results['base_emb_sim']

    # include segment-level embedding deltas if available
    seg_lines = []
    for seg in ["seg_instruction","seg_input","seg_asst_prefix"]:
        b = results['base_emb_seg_sims'].get(seg, None)
        f = results['ft_emb_seg_sims'].get(seg, None)
        if b is not None and f is not None:
            seg_lines.append(f"| {seg} | `{b:.6f}` | `{f:.6f}` | `{(f-b):+.6f}` |")

    report = [
        f"# Comprehensive Analysis Report: Case Study",
        f"- **Prompt ID:** `{prompt_id}`",
        f"- **Paraphrase Type:** `{p_key}`",
        "---",
        "## 1. Embedding Space Analysis",
        "| Statistic | Value |",
        "|---|---|",
        f"| Base Model Embedding Similarity (masked) | `{results['base_emb_sim']:.6f}` |",
        f"| Fine-Tuned Model Embedding Similarity (masked) | `{results['ft_emb_sim']:.6f}` |",
        f"| **Delta (FT - Base)** | **`{emb_sim_delta:+.6f}`** |",
    ]
    if seg_lines:
        report += [
            "",
            "### Segment-wise Embedding Similarity",
            "| Segment | Base | FT | Δ |",
            "|---|---|---|---|",
            *seg_lines
        ]

    report += [
        "---",
        "## 2. Representational Similarity (Hidden States)",
        "| Statistic | Value |",
        "|---|---|",
        f"| Mean Δ (FT - Base) | `{sim_delta.mean():.6f}` |",
        f"| Paired T-test p-value | `{sim_ttest.pvalue:.6f}` |",
        f"| **Significant?** | **{'YES' if sim_ttest.pvalue < 0.05 else 'NO'}** |",
        "---",
        "## 3. Activation Norm Analysis (Hidden States, Paraphrase, masked means)",
        "| Statistic | Value |",
        "|---|---|",
        f"| Mean Δ | `{norm_delta.mean():.6f}` |",
        f"| Paired T-test p-value | `{norm_ttest.pvalue:.6f}` |",
        f"| **Significant?** | **{'YES' if norm_ttest.pvalue < 0.05 else 'NO'}** |",
        "---",
        "## 4. Raw Data by Layer",
        "| Layer | Base Sim. | FT Sim. | Sim Δ | Base Norm | FT Norm | Norm Δ |",
        "|---|---|---|---|---|---|---|"
    ]
    for i in results['base_cos_sims'].keys():
        report.append(f"| {i:<5} | {results['base_cos_sims'][i]:.6f} | {results['ft_cos_sims'][i]:.6f} | {(results['ft_cos_sims'][i]-results['base_cos_sims'][i]):+.6f} | {results['base_norms_para'][i]:.6f} | {results['ft_norms_para'][i]:.6f} | {(results['ft_norms_para'][i]-results['base_norms_para'][i]):+.6f} |")
    (output_path / f"report_case_study_prompt_{prompt_id}_{p_key}.md").write_text("\n".join(report), encoding="utf-8")

def report_aggregate_numerics(agg_data: Dict, num_samples: int, output_path: Path):
    if not any(agg_data.values()): return
    all_dfs = {
        "Embedding Similarity": pd.DataFrame([(p, v, 'base') for p, V in agg_data['base_emb_sims'].items() for v in V] + 
                                            [(p, v, 'ft') for p, V in agg_data['ft_emb_sims'].items() for v in V], 
                                            columns=['paraphrase', 'value', 'model']),
        "Representational Similarity": pd.DataFrame([(p, l, v, 'base') for p, L in agg_data['base_sims'].items() for l, V in L.items() for v in V] +
                                                    [(p, l, v, 'ft') for p, L in agg_data['ft_sims'].items() for l, V in L.items() for v in V],
                                                    columns=['paraphrase', 'layer', 'value', 'model']),
        "Activation Norm": pd.DataFrame([(p, l, v, 'base') for p, L in agg_data['base_norms'].items() for l, V in L.items() for v in V] +
                                        [(p, l, v, 'ft') for p, L in agg_data['ft_norms'].items() for l, V in L.items() for v in V],
                                        columns=['paraphrase', 'layer', 'value', 'model'])
    }
    report_md = ["# Aggregate Numerical Analysis Report", f"Based on **{num_samples}** processed prompts.", "---"]
    report_json = {"num_samples": num_samples, "analyses": {}}
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

# Scores Loader + Correlations
def load_scores(scores_path: str) -> Dict[Tuple[int, str], float]:
    """
    Returns mapping (prompt_count, paraphrase_key) -> TF score (index 0).
    Also includes ('instruction_original') as a key.
    """
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
    """
    item_rows: one row per (pid, pkey) with many Δmetrics.
    Computes Pearson and Spearman correlations with ΔTF (FT-Base).
    """
    if not scores_base or not scores_ft:
        script_logger.info("Scores not provided for correlations; skipping.")
        return
    records = []
    for row in item_rows:
        pid, pkey = row["prompt_count"], row["paraphrase_key"]
        if (pid, pkey) in scores_base and (pid, pkey) in scores_ft:
            delta_tf = scores_ft[(pid, pkey)] - scores_base[(pid, pkey)]
            rec = {"prompt_count": pid, "paraphrase_key": pkey, "delta_tf": delta_tf}
            rec.update(row)  # includes metric deltas
            records.append(rec)
    if not records:
        script_logger.warning("No overlapping score entries for correlation.")
        return
    df = pd.DataFrame(records)

    # Collect candidate metrics to correlate with ΔTF
    metric_cols = [c for c in df.columns if c.startswith("delta_") and c not in {"delta_tf"}]
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

    # Also dump df for per-item rows
    df.to_csv(output_path / "per_item_metrics_with_deltaTF.csv", index=False)
    script_logger.info("Saved correlations with ΔTF.")

# Main impl
def main():
    parser = argparse.ArgumentParser(description="Advanced Robustness Analyzer for LLM Activations (extended).", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--ft_model_path", type=str, required=True)
    parser.add_argument("--prompts_json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="robustness_analysis_results")
    parser.add_argument("--run_mode", type=str, choices=['case_study', 'aggregate'], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--paraphrase_keys", type=str, nargs='+')
    parser.add_argument("--prompt_ids", type=int, nargs='+')
    parser.add_argument("--data_split", type=str, choices=['all', 'train', 'val', 'test'], default='test')
    parser.add_argument("--val_size", type=float, default=0.05)
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    # options
    parser.add_argument("--compute_pii", action="store_true", help="Compute Paraphrase Invariance Index per layer (Base & FT).")
    parser.add_argument("--compute_svd", action="store_true", help="Compute SVD summaries for 2D weight deltas.")
    parser.add_argument("--scores_base_json_path", type=str, default="", help="Scores JSON for base (TF at index 0).")
    parser.add_argument("--scores_ft_json_path", type=str, default="", help="Scores JSON for FT (TF at index 0).")

    args = parser.parse_args()

    output_path = Path(args.output_dir); output_path.mkdir(exist_ok=True, parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu": script_logger.warning("CUDA not available. Running on CPU will be extremely slow.")

    script_logger.info("--- Model and Data Loading ---")
    base_model, tokenizer = load_model_and_tokenizer(args.base_model_path, args.base_model_path, device)
    ft_model, _ = load_model_and_tokenizer(args.ft_model_path, args.base_model_path, device)

    with open(args.prompts_json_path, 'r', encoding='utf-8') as f: prompts_data_all = json.load(f)

    if args.data_split != 'all':
        all_pids = [item['prompt_count'] for item in prompts_data_all if 'prompt_count' in item]
        train_ids, val_ids, test_ids = get_group_wise_split_ids(all_pids, args.val_size, args.test_size, args.seed)
        target_ids = {'train': train_ids, 'val': val_ids, 'test': test_ids}[args.data_split]
        prompts_data = [item for item in prompts_data_all if item.get('prompt_count') in target_ids]
    else:
        prompts_data = prompts_data_all
    
    if args.limit > 0 and args.run_mode == 'aggregate':
        prompts_data = prompts_data[:args.limit]
        script_logger.info(f"Limited aggregate run to the first {len(prompts_data)} items.")
    prompts_map = {item['prompt_count']: item for item in prompts_data}

    # Load scores (optional)
    scores_base = load_scores(args.scores_base_json_path) if args.scores_base_json_path else {}
    scores_ft   = load_scores(args.scores_ft_json_path)   if args.scores_ft_json_path   else {}

    if args.run_mode == 'aggregate':
        # Detailed ΔW (keeps legacy plot and adds CSVs + bucket plots)
        analyze_and_plot_weight_deltas(base_model, ft_model, output_path, compute_svd=args.compute_svd)

        aggregate_data = {
            'base_emb_sims': defaultdict(list), 'ft_emb_sims': defaultdict(list),
            'base_sims': defaultdict(lambda: defaultdict(list)), 'ft_sims': defaultdict(lambda: defaultdict(list)),
            'base_norms': defaultdict(lambda: defaultdict(list)), 'ft_norms': defaultdict(lambda: defaultdict(list))
        }
        # segment-level embeddings & norms, PII, and per-item metric rows for correlations
        seg_emb_base, seg_emb_ft = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
        seg_norm_base, seg_norm_ft = defaultdict(lambda: defaultdict(lambda: defaultdict(list))), defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        pii_base_per_layer, pii_ft_per_layer = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
        per_item_rows_for_corr = []

        # Outlier tracking via robust z on Δ embedding similarity (global)
        outlier_records = []
        emb_sim_deltas_for_mad = []

        # Precompute unrelated mapping (simple ring shift of originals)
        originals = [(it['prompt_count'], it.get('instruction_original', '')) for it in prompts_data if it.get('instruction_original')]
        unrelated_lookup = {}
        for i, (pid, instr) in enumerate(originals):
            j = (i + 1) % len(originals)
            unrelated_lookup[pid] = originals[j][1]

        num_processed = 0
        for i, prompt_item in enumerate(prompts_data):
            pid, original_text = prompt_item.get('prompt_count'), prompt_item.get('instruction_original')
            inp = prompt_item.get('input', '')
            if not all([pid, original_text]): continue
            keys_to_process = args.paraphrase_keys if args.paraphrase_keys else [k for k in prompt_item if k.startswith('instruct_') and prompt_item[k]]
            for p_key in keys_to_process:
                paraphrase_text = prompt_item.get(p_key)
                if not paraphrase_text: continue
                script_logger.info(f"Processing... [Prompt {i+1}/{len(prompts_data)}] [ID: {pid}] [{p_key}]")
                try:
                    results = run_and_analyze_pair(base_model, ft_model, tokenizer, original_text, paraphrase_text, inp, device)

                    # Outlier tracking (record Δ after we know MAD scale)
                    emb_sim_delta = results['ft_emb_sim'] - results['base_emb_sim']
                    emb_sim_deltas_for_mad.append(emb_sim_delta)

                    # Collect aggregates
                    aggregate_data['base_emb_sims'][p_key].append(results['base_emb_sim'])
                    aggregate_data['ft_emb_sims'][p_key].append(results['ft_emb_sim'])
                    for layer_idx in results['base_cos_sims']:
                        aggregate_data['base_sims'][p_key][layer_idx].append(results['base_cos_sims'][layer_idx])
                        aggregate_data['ft_sims'][p_key][layer_idx].append(results['ft_cos_sims'][layer_idx])
                        aggregate_data['base_norms'][p_key][layer_idx].append(results['base_norms_para'][layer_idx])
                        aggregate_data['ft_norms'][p_key][layer_idx].append(results['ft_norms_para'][layer_idx])

                    # Segment-level embedding sims (if present)
                    for seg in ["seg_instruction","seg_input","seg_asst_prefix"]:
                        b = results['base_emb_seg_sims'].get(seg, None)
                        f = results['ft_emb_seg_sims'].get(seg, None)
                        if b is not None: seg_emb_base[p_key][seg].append(b)
                        if f is not None: seg_emb_ft[p_key][seg].append(f)

                    # Segment-level norms (paraphrase)
                    for seg in ["seg_instruction","seg_input","seg_asst_prefix"]:
                        bnorms = results['base_norms_para_segs'].get(seg, {})
                        fnorms = results['ft_norms_para_segs'].get(seg, {})
                        for l, v in bnorms.items(): seg_norm_base[p_key][seg][l].append(v)
                        for l, v in fnorms.items(): seg_norm_ft[p_key][seg][l].append(v)

                    # PII (optional, Base & FT)
                    if args.compute_pii:
                        unrelated_instr = unrelated_lookup.get(pid, original_text)
                        pii_b = compute_pii_for_item(base_model, tokenizer, original_text, paraphrase_text, unrelated_instr, inp, device)
                        pii_f = compute_pii_for_item(ft_model,   tokenizer, original_text, paraphrase_text, unrelated_instr, inp, device)
                        for l in pii_b: pii_base_per_layer[p_key][l].append(pii_b[l])
                        for l in pii_f: pii_ft_per_layer[p_key][l].append(pii_f[l])

                    # Rows for correlation with ΔTF (final-layer deltas + embedding/global deltas)
                    # Choose convenient layer summary: final layer index
                    final_layer = max(results['base_cos_sims'].keys())
                    row = {
                        "prompt_count": pid,
                        "paraphrase_key": p_key,
                        "delta_emb_sim_all": emb_sim_delta,
                        "delta_cos_sim_final": results['ft_cos_sims'][final_layer] - results['base_cos_sims'][final_layer],
                        "delta_cos_sim_mean": float(np.mean(list(results['ft_cos_sims'].values()))) - float(np.mean(list(results['base_cos_sims'].values()))),
                        "delta_norm_mean": float(np.mean(list(results['ft_norms_para'].values()))) - float(np.mean(list(results['base_norms_para'].values()))),
                    }
                    # add segment deltas if present
                    for seg in ["seg_instruction","seg_input","seg_asst_prefix"]:
                        b = results['base_emb_seg_sims'].get(seg, None)
                        f = results['ft_emb_seg_sims'].get(seg, None)
                        if b is not None and f is not None:
                            row[f"delta_emb_sim_{seg}"] = f - b
                    per_item_rows_for_corr.append(row)

                except Exception as e:
                    script_logger.error(f"Failed on Prompt ID {pid}, Paraphrase '{p_key}'. Error: {e}")
            num_processed += 1
            
        # Robust outliers (MAD z)
        if emb_sim_deltas_for_mad:
            arr = np.array(emb_sim_deltas_for_mad)
            z = robust_mad_z(arr)
            THRESH = 3.5  # common robust threshold
            z_idx = 0
            for i, prompt_item in enumerate(prompts_data):
                pid = prompt_item.get('prompt_count')
                keys_to_process = args.paraphrase_keys if args.paraphrase_keys else [k for k in prompt_item if k.startswith('instruct_') and prompt_item[k]]
                for p_key in keys_to_process:
                    if z_idx >= len(z): break
                    if abs(z[z_idx]) >= THRESH:
                        outlier_records.append({"pid": pid, "p_key": p_key, "mad_z": float(z[z_idx])})
                    z_idx += 1

        if num_processed > 0:
            script_logger.info(f"Aggregation complete. Processed {num_processed} prompts.")

            # Write outlier report (MAD-based)
            if outlier_records:
                report_path = output_path / "embedding_similarity_outliers_mad.md"
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("# Embedding Similarity Outlier Report (MAD z-score)\n\n")
                    f.write("Entries where |MAD z| ≥ 3.5\n\n")
                    for r in outlier_records:
                        f.write(f"- Prompt ID `{r['pid']}` | Paraphrase `{r['p_key']}` | MAD z = `{r['mad_z']:+.3f}`\n")
                script_logger.info(f"Wrote robust outlier report: {report_path}")

            # Legacy plots + aggregate numeric reports
            plot_aggregate_results(aggregate_data, num_processed, output_path)
            report_aggregate_numerics(aggregate_data, num_processed, output_path)

            # Save aggregates to CSVs (segments + PII)
            # Segment embedding sims
            seg_rows = []
            for pkey, segmap in seg_emb_base.items():
                for seg, vals in segmap.items():
                    for v in vals:
                        seg_rows.append({"paraphrase": pkey, "segment": seg, "model": "base", "value": v})
            for pkey, segmap in seg_emb_ft.items():
                for seg, vals in segmap.items():
                    for v in vals:
                        seg_rows.append({"paraphrase": pkey, "segment": seg, "model": "ft", "value": v})
            if seg_rows:
                pd.DataFrame(seg_rows).to_csv(output_path / "segment_embedding_similarities.csv", index=False)

            # Segment norms per layer (paraphrase)
            seg_norm_rows = []
            for pkey, segmap in seg_norm_base.items():
                for seg, layermap in segmap.items():
                    for l, vals in layermap.items():
                        for v in vals:
                            seg_norm_rows.append({"paraphrase": pkey, "segment": seg, "layer": l, "model": "base", "value": v})
            for pkey, segmap in seg_norm_ft.items():
                for seg, layermap in segmap.items():
                    for l, vals in layermap.items():
                        for v in vals:
                            seg_norm_rows.append({"paraphrase": pkey, "segment": seg, "layer": l, "model": "ft", "value": v})
            if seg_norm_rows:
                pd.DataFrame(seg_norm_rows).to_csv(output_path / "segment_norms_paraphrase_per_layer.csv", index=False)

            # PII
            if args.compute_pii:
                pii_rows = []
                for pkey, layermap in pii_base_per_layer.items():
                    for l, vals in layermap.items():
                        for v in vals:
                            pii_rows.append({"paraphrase": pkey, "layer": l, "model": "base", "pii": v})
                for pkey, layermap in pii_ft_per_layer.items():
                    for l, vals in layermap.items():
                        for v in vals:
                            pii_rows.append({"paraphrase": pkey, "layer": l, "model": "ft", "pii": v})
                if pii_rows:
                    pd.DataFrame(pii_rows).to_csv(output_path / "pii_per_layer.csv", index=False)

            # Correlations with ΔTF (optional)
            compute_correlations(per_item_rows_for_corr, scores_base, scores_ft, output_path)

        else:
            script_logger.warning("No prompts were successfully processed.")

    elif args.run_mode == 'case_study':
        pids_to_process = []
        if args.prompt_ids: pids_to_process = [pid for pid in args.prompt_ids if pid in prompts_map]
        elif args.limit > 0: pids_to_process = [item['prompt_count'] for item in prompts_data[:args.limit]]
        for pid in pids_to_process:
            item = prompts_map[pid]
            keys_to_process = args.paraphrase_keys if args.paraphrase_keys else [k for k in item if k.startswith('instruct_')]
            for p_key in keys_to_process:
                original_text, paraphrase_text = item.get('instruction_original'), item.get(p_key)
                inp = item.get('input', '')
                if not all([original_text, paraphrase_text]): continue
                script_logger.info(f"Analyzing Prompt ID: {pid}, Paraphrase: {p_key}")
                try:
                    results = run_and_analyze_pair(base_model, ft_model, tokenizer, original_text, paraphrase_text, inp, device)
                    plot_case_study(results, pid, p_key, output_path)
                    report_case_study_numerics(results, pid, p_key, output_path)
                except Exception as e:
                    script_logger.error(f"Failed on Prompt ID {pid}, Paraphrase '{p_key}'. Error: {e}"); traceback.print_exc()

    script_logger.info(f"Analysis complete. All outputs saved to: {output_path}")

if __name__ == "__main__":
    main()
