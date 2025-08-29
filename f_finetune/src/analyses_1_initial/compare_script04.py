
#!/usr/bin/env python3
"""
compare_script04_extra.py

Purpose
-------
A) Diff-of-diffs (DoD) at three residual stream points for each layer:
   - resid_pre  : input to the layer, before attention
   - resid_mid  : after attention add (resid_pre + attn_out)
   - resid_post : after MLP add (layer output)

B) Identify "collapse" layers (where the Base distance between orig↔para exceeds FT distance),
   and attribute changes to downstream weight matrices (q/k/v/o, up/gate/down).

C) For the most culpable layer+matrix, run PCA/SVD (on the matrix) pre- vs post-finetune
   and compare spectra & cumulative explained variance. Optionally, project activations
   through the matrix and analyze variance collapse.

Usage (example)----
python compare_script04.py \
  --base_model_path f_finetune/model \
  --ft_model_path f_finetune/outputs/.../final \
  --prompts_json_path a_data/alpaca/50k_phrxed.json \
  --output_dir f_finetune/outputs/.../extra_suite \
  --dtype bfloat16 \
  --limit 200 \
  --pairs_per_batch 8 \
  --focus_on_answer_tokens \
  --capture_attn_qkv --capture_mlp_up_gate \
  --flash_attention auto \
  --run_mode extra_suite

Notes
-----
- We reconstruct resid_mid as resid_pre + attn_o_out. We capture attn_o_out (the post-o_proj output).
- resid_post is the layer forward output (same as in your prior ActivationExtractor).
- Works with Llama/Gemma-like decoder architectures in Hugging Face Transformers.
"""
import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

# =========================
# Logging
# =========================
class SafeFormatter(logging.Formatter):
    def format(self, record):
        rec = logging.makeLogRecord(record.__dict__)
        if not isinstance(rec.msg, str):
            try:
                rec.msg = str(rec.msg)
            except Exception:
                rec.msg = "<non-string log message>"
        return super().format(rec)

log = logging.getLogger("extra")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(SafeFormatter("[%(levelname)s] %(message)s"))
log.addHandler(_handler)

# =========================
# Helpers
# =========================
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.device).float()
    denom = m.sum().clamp_min(1.0)
    return (x * m.unsqueeze(-1)).sum(dim=0) / denom

def module_bucket(name: str) -> Tuple[str, str, int]:
    lname = name.lower()
    layer_idx = -1
    key = None
    parts = lname.split("layers.")
    if len(parts) > 1:
        try:
            layer_part = parts[1]
            layer_idx = int(layer_part.split(".", 1)[0])
        except Exception:
            pass
    if ".self_attn." in lname or ".attention." in lname:
        scope = "attn"
        if ".q_proj" in lname: key = "q"
        elif ".k_proj" in lname: key = "k"
        elif ".v_proj" in lname: key = "v"
        elif ".o_proj" in lname: key = "o"
        else: key = "attn_misc"
    elif ".mlp." in lname or ".feed_forward." in lname:
        scope = "mlp"
        if ".up_proj" in lname: key = "up"
        elif ".gate_proj" in lname: key = "gate"
        elif ".down_proj" in lname: key = "down"
        else: key = "mlp_misc"
    else:
        scope = "other"; key = "other"
    return scope, key, layer_idx

# ---------- model/layer introspection that works with and without LoRA ----------
def _maybe_get_base_model(model: nn.Module) -> nn.Module:
    # For PEFT, get the wrapped base; otherwise return model
    if PeftModel is not None and isinstance(model, PeftModel):
        if hasattr(model, "get_base_model"):
            try:
                return model.get_base_model()
            except Exception:
                pass
        if hasattr(model, "base_model"):
            try:
                return model.base_model
            except Exception:
                pass
    return model

def _find_decoder_layers(model: nn.Module) -> List[nn.Module]:
    """
    Locate the list of transformer decoder layers robustly.
    Works for Gemma/LLaMA-like models and when wrapped by PEFT.
    """
    m = _maybe_get_base_model(model)
    # Common path: m.model.layers is a ModuleList
    if hasattr(m, "model") and hasattr(m.model, "layers") and isinstance(m.model.layers, nn.ModuleList):
        return list(m.model.layers)

    # Fallback: search any submodule that has a 'layers' ModuleList
    for _, mod in m.named_modules():
        if hasattr(mod, "layers") and isinstance(mod.layers, nn.ModuleList):
            return list(mod.layers)

    # Last resort: collect modules that look like decoder layers (have self_attn and mlp)
    candidates = []
    for _, mod in m.named_modules():
        if hasattr(mod, "self_attn") and hasattr(mod, "mlp"):
            candidates.append(mod)
    if candidates:
        return candidates

    raise TypeError("Could not locate decoder layers for hooks; model structure not recognized.")

# =========================
# Tokenisation with segment masks
# =========================
@dataclass
class EncodedExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    seg_instruction: torch.Tensor
    seg_input: torch.Tensor
    seg_asst_prefix: torch.Tensor
    seg_answer: torch.Tensor

def tokenise_with_segments(tokenizer, instruction: str, inp: str, device: str,
                           answer_text: Optional[str] = None) -> EncodedExample:
    user_msg = instruction if not inp else f"{instruction}\n\nInput: {inp}"
    msgs = [{"role":"user","content":user_msg}, {"role":"assistant","content":""}]
    asst_ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )
    seg_len_asst = asst_ids.shape[1]
    instr_only = tokenizer.apply_chat_template(
        [{"role":"user","content": instruction}], tokenize=True,
        add_generation_prompt=False, return_tensors="pt"
    )
    len_instr_only = instr_only.shape[1]
    if inp:
        user_w_input = tokenizer.apply_chat_template(
            [{"role":"user","content": user_msg}], tokenize=True,
            add_generation_prompt=False, return_tensors="pt"
        )
        len_user_w_input = user_w_input.shape[1]
        input_len = max(0, len_user_w_input - len_instr_only)
    else:
        input_len = 0

    if answer_text:
        ans_ids = tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        input_ids = torch.cat([asst_ids, ans_ids], dim=1)
    else:
        input_ids = asst_ids

    attention_mask = torch.ones_like(input_ids)
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

    seg_asst_prefix[min(seg_len_asst, total_len)-1:min(seg_len_asst, total_len)] = True
    if total_len > seg_len_asst:
        seg_answer[seg_len_asst:total_len] = True

    return EncodedExample(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        seg_instruction=seg_instruction.to(device),
        seg_input=seg_input.to(device),
        seg_asst_prefix=seg_asst_prefix.to(device),
        seg_answer=seg_answer.to(device),
    )

def pad_and_stack(
    examples: List[EncodedExample],
    pad_token_id: int,
    padding_side: str = "left",
) -> Dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("pad_and_stack called with empty list")

    max_len = max(int(e.input_ids.shape[1]) for e in examples)

    def _pad_2d(x: torch.Tensor, pad_val: int):
        diff = max_len - int(x.shape[1])
        if diff <= 0:
            return x
        if padding_side == "left":
            return F.pad(x, (diff, 0), value=pad_val)
        else:
            return F.pad(x, (0, diff), value=pad_val)

    def _pad_1d_bool(x: torch.Tensor):
        diff = max_len - int(x.shape[0])
        if diff <= 0:
            return x
        if padding_side == "left":
            return F.pad(x.to(torch.int8), (diff, 0), value=0).to(torch.bool)
        else:
            return F.pad(x.to(torch.int8), (0, diff), value=0).to(torch.bool)

    inputs, attn = [], []
    seg_inst, seg_inp, seg_pref, seg_ans = [], [], [], []

    for e in examples:
        inputs.append(_pad_2d(e.input_ids, pad_token_id))
        attn.append(_pad_2d(e.attention_mask, 0))
        seg_inst.append(_pad_1d_bool(e.seg_instruction))
        seg_inp.append(_pad_1d_bool(e.seg_input))
        seg_pref.append(_pad_1d_bool(e.seg_asst_prefix))
        seg_ans.append(_pad_1d_bool(e.seg_answer))

    input_ids = torch.cat(inputs, dim=0)
    attention_mask = torch.cat(attn, dim=0)
    seg_instruction = torch.stack(seg_inst, dim=0)
    seg_input       = torch.stack(seg_inp, dim=0)
    seg_asst_prefix = torch.stack(seg_pref, dim=0)
    seg_answer      = torch.stack(seg_ans, dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "seg_instruction": seg_instruction,
        "seg_input": seg_input,
        "seg_asst_prefix": seg_asst_prefix,
        "seg_answer": seg_answer,
    }

# =========================
# Probes
# =========================
class LayerIOProbe:
    """
    Captures per-layer:
      - layer_input (resid_pre)
      - attn_o (self_attn.o_proj output)
      - mlp_down (mlp.down_proj output)
      - layer_output (resid_post)
    resid_mid is reconstructed as layer_input + attn_o.

    NOTE: We introspect decoder layers so this works for both bare and PEFT-wrapped models.
    """
    def __init__(self, model: PreTrainedModel,
                 capture_attn_qkv: bool = False,
                 capture_mlp_up_gate: bool = False):
        self.model = model
        self.layer_input: Dict[int, torch.Tensor] = {}
        self.layer_output: Dict[int, torch.Tensor] = {}
        self.attn_o: Dict[int, torch.Tensor] = {}
        self.mlp_down: Dict[int, torch.Tensor] = {}
        self.q: Dict[int, torch.Tensor] = {} if capture_attn_qkv else None
        self.k: Dict[int, torch.Tensor] = {} if capture_attn_qkv else None
        self.v: Dict[int, torch.Tensor] = {} if capture_attn_qkv else None
        self.up: Dict[int, torch.Tensor] = {} if capture_mlp_up_gate else None
        self.gate: Dict[int, torch.Tensor] = {} if capture_mlp_up_gate else None
        self._handles = []
        self.capture_attn_qkv = capture_attn_qkv
        self.capture_mlp_up_gate = capture_mlp_up_gate
        self._layers: List[nn.Module] = []

    def _hook_store(self, store: Dict[int, torch.Tensor], idx: int):
        def hook(_, __, out):
            try:
                if isinstance(out, (tuple, list)):
                    out = out[0]
                store[idx] = out.detach()
            except Exception:
                pass
        return hook

    def _hook_input(self, idx: int):
        # Robust to both (module, inputs) and (module, inputs, kwargs)
        def hook(*args):
            try:
                if len(args) == 2:
                    _, inp = args
                elif len(args) >= 3:
                    _, inp, _kwargs = args[0], args[1], args[2]
                    inp = args[1]
                else:
                    return
                x = inp[0] if isinstance(inp, (tuple, list)) and len(inp) > 0 else inp
                if isinstance(x, (tuple, list)) and len(x) > 0:
                    x = x[0]
                if isinstance(x, (tuple, list)):
                    for el in x:
                        if torch.is_tensor(el):
                            x = el
                            break
                if torch.is_tensor(x):
                    self.layer_input[idx] = x.detach()
            except Exception:
                pass
        return hook

    def _hook_output(self, idx: int):
        def hook(_, __, out):
            try:
                if isinstance(out, tuple):
                    out = out[0]
                self.layer_output[idx] = out.detach()
            except Exception:
                pass
        return hook

    def __enter__(self):
        self.layer_input.clear(); self.layer_output.clear()
        self.attn_o.clear(); self.mlp_down.clear()
        if self.q is not None: self.q.clear(); self.k.clear(); self.v.clear()
        if self.up is not None: self.up.clear(); self.gate.clear()

        # Find decoder layer modules robustly (works w/ PEFT too)
        try:
            self._layers = _find_decoder_layers(self.model)
        except Exception as e:
            raise TypeError(f"Model layer discovery failed: {e}")

        for i, layer in enumerate(self._layers):
            self._handles.append(layer.register_forward_pre_hook(self._hook_input(i)))
            self._handles.append(layer.register_forward_hook(self._hook_output(i)))
            for name, mod in layer.named_modules():
                if name.endswith("o_proj"):
                    self._handles.append(mod.register_forward_hook(self._hook_store(self.attn_o, i)))
                if name.endswith("down_proj"):
                    self._handles.append(mod.register_forward_hook(self._hook_store(self.mlp_down, i)))
                if self.capture_attn_qkv:
                    if name.endswith("q_proj"):
                        self._handles.append(mod.register_forward_hook(self._hook_store(self.q, i)))
                    if name.endswith("k_proj"):
                        self._handles.append(mod.register_forward_hook(self._hook_store(self.k, i)))
                    if name.endswith("v_proj"):
                        self._handles.append(mod.register_forward_hook(self._hook_store(self.v, i)))
                if self.capture_mlp_up_gate:
                    if name.endswith("up_proj"):
                        self._handles.append(mod.register_forward_hook(self._hook_store(self.up, i)))
                    if name.endswith("gate_proj"):
                        self._handles.append(mod.register_forward_hook(self._hook_store(self.gate, i)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        return False

# =========================
# Data & Models
# =========================
def load_prompts(path: str, limit: Optional[int] = None) -> List[dict]:
    data = json.loads(Path(path).read_text())
    if limit is not None:
        data = data[:limit]
    return data

def load_model_and_tokenizer(model_path: str, base_path_for_tokenizer: str, device: str,
                             dtype_str: str = "bfloat16"):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]
    tok = AutoTokenizer.from_pretrained(base_path_for_tokenizer, use_fast=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    is_peft = False
    peft_loaded_ok = False
    model = None

    if PeftModel is not None:
        try:
            base = AutoModelForCausalLM.from_pretrained(base_path_for_tokenizer, torch_dtype=dtype, device_map=device)
            model = PeftModel.from_pretrained(base, model_path, torch_dtype=dtype)
            is_peft = True
            peft_loaded_ok = True
            log.info("Loaded PEFT/LoRA adapter from %s", model_path)
        except Exception:
            model = None

    if model is None:
        # Either not a PEFT adapter, or PEFT not available: load as a full model
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)
        log.info("Loaded full model weights from %s", model_path)

    model.eval()
    info = {"is_peft": is_peft and peft_loaded_ok, "model_path": model_path, "base_path": base_path_for_tokenizer, "dtype": dtype_str, "device": device}
    return model, tok, info

def build_effective_ft_model_for_weights(ft_info: dict) -> Optional[PreTrainedModel]:
    """
    Return a CPU model whose weights equal the effective fine-tuned weights:
      - If LoRA/PEFT: base + merged adapters (merge_and_unload) on CPU
      - Else: None (meaning use the already-loaded FT model for weights)
    """
    if not ft_info.get("is_peft", False):
        return None
    if PeftModel is None:
        return None

    base_path = ft_info["base_path"]
    adapter_path = ft_info["model_path"]
    # Load on CPU to avoid VRAM use for weight diffs/PCA
    try:
        base_cpu = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16, device_map="cpu")
        ft_cpu = PeftModel.from_pretrained(base_cpu, adapter_path, torch_dtype=torch.float16, device_map="cpu")
        ft_cpu = ft_cpu.merge_and_unload()  # now a plain base model with merged weights
        ft_cpu.eval()
        log.info("Built effective merged FT model on CPU for weight diffs/PCA.")
        return ft_cpu
    except Exception as e:
        log.warning("Failed to build merged FT model on CPU: %s", e)
        return None

# =========================
# Core: diff-of-diffs on residual stream
# =========================
def segment_mask_from_batch(batch: Dict[str, torch.Tensor], focus_on_answer: bool) -> torch.Tensor:
    if focus_on_answer and batch["seg_answer"].any():
        return batch["seg_answer"].bool()
    return batch["attention_mask"].bool()

def reps_from_probe(probe: LayerIOProbe, idx_item: int, mask_2d: torch.Tensor) -> Dict[str, Dict[int, torch.Tensor]]:
    reps = {"resid_pre":{}, "resid_mid":{}, "resid_post":{}}
    all_layers = sorted(set(probe.layer_input.keys()) | set(probe.layer_output.keys()))
    for l in all_layers:
        pre  = probe.layer_input.get(l)
        post = probe.layer_output.get(l)
        attno = probe.attn_o.get(l)
        if pre is None or post is None or attno is None:
            continue
        mask = mask_2d[idx_item]
        pre_vec  = masked_mean(pre[idx_item],  mask)
        mid_vec  = masked_mean((pre[idx_item] + attno[idx_item]), mask)
        post_vec = masked_mean(post[idx_item], mask)
        reps["resid_pre"][l]  = pre_vec
        reps["resid_mid"][l]  = mid_vec
        reps["resid_post"][l] = post_vec
    return reps

def pairwise_layer_distances(reps_a: Dict[int, torch.Tensor],
                             reps_b: Dict[int, torch.Tensor]) -> Dict[int, float]:
    common = sorted(set(reps_a.keys()) & set(reps_b.keys()))
    d = {}
    for l in common:
        va, vb = reps_a[l], reps_b[l]
        d[l] = torch.linalg.vector_norm((va - vb).float()).item()
    return d

def run_resid_dod(base_model, ft_model, tokenizer, prompts: List[dict], device: str,
                  output_dir: Path, focus_on_answer: bool = False,
                  pairs_per_batch: int = 8, limit: Optional[int] = None):
    """
    Compute diff-of-diffs at three residual sites:
      - resid_pre  (layer input, before attention and MLP)
      - resid_mid  (after attention output is added)
      - resid_post (layer output, after MLP)
    For each layer/site, we measure L2 distance between reps of (original instruction vs paraphrase)
    on base vs fine-tuned, then take: (Base distance − FT distance).
    """
    out_dir = output_dir / "resid_dod"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    def paraphrase_keys(d):
        return [k for k in d.keys() if k.startswith("instruct_") and k != "instruction_original"]

    selected = prompts if limit is None else prompts[:limit]

    for batch_start in range(0, len(selected), pairs_per_batch):
        batch = selected[batch_start: batch_start + pairs_per_batch]

        encs_o, encs_p, chosen_keys = [], [], []
        for ex in batch:
            instr = ex.get("instruction_original","")
            inp   = ex.get("input","") or ""
            keys = paraphrase_keys(ex)
            if not keys:
                continue
            pkey = keys[0]
            para = ex.get(pkey, "")
            encs_o.append(tokenise_with_segments(tokenizer, instr, inp, device, answer_text=None))
            encs_p.append(tokenise_with_segments(tokenizer, para,  inp, device, answer_text=None))
            chosen_keys.append(pkey)

        if not encs_o:
            continue

        # Base pass
        pad_side = getattr(tokenizer, "padding_side", "left")
        base_batch = pad_and_stack(encs_o + encs_p, tokenizer.pad_token_id, padding_side=pad_side)
        with LayerIOProbe(base_model) as probe:
            _ = base_model(
                base_batch["input_ids"],
                attention_mask=base_batch["attention_mask"],
                use_cache=True
            )
        masks_all = segment_mask_from_batch(base_batch, focus_on_answer)

        reps_items = []
        for i in range(base_batch["input_ids"].shape[0]):
            reps_items.append(reps_from_probe(probe, i, masks_all))

        # Pair orig i with para i
        B = len(encs_o)
        for i in range(B):
            reps_o = reps_items[i]
            reps_p = reps_items[i + B]
            for where in ["resid_pre","resid_mid","resid_post"]:
                d_base = pairwise_layer_distances(reps_o[where], reps_p[where])

                ft_pair = pad_and_stack([encs_o[i], encs_p[i]], tokenizer.pad_token_id, padding_side=pad_side)
                with LayerIOProbe(ft_model) as probe_ft:
                    _ = ft_model(
                        ft_pair["input_ids"],
                        attention_mask=ft_pair["attention_mask"],
                        use_cache=True
                    )
                masks_ft = segment_mask_from_batch(ft_pair, focus_on_answer)
                reps_o_ft = reps_from_probe(probe_ft, 0, masks_ft)
                reps_p_ft = reps_from_probe(probe_ft, 1, masks_ft)
                d_ft = pairwise_layer_distances(reps_o_ft[where], reps_p_ft[where])

                common_layers = sorted(set(d_base.keys()) & set(d_ft.keys()))
                for l in common_layers:
                    rows.append({
                        "prompt_count": batch[i].get("prompt_count", -1),
                        "paraphrase_key": chosen_keys[i],
                        "layer": l,
                        "where": where,
                        "l2_base": d_base[l],
                        "l2_ft": d_ft[l],
                        "diff_of_diffs_l2": d_base[l] - d_ft[l],
                    })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "resid_diff_of_diffs_by_layer.csv", index=False)

    if not df.empty:
        for where in ["resid_pre","resid_mid","resid_post"]:
            sub = df[df["where"] == where]
            grp = sub.groupby("layer")[["l2_base","l2_ft","diff_of_diffs_l2"]].mean().reset_index()

            plt.figure()
            plt.plot(grp["layer"], grp["l2_base"], label="Base L2 (orig↔para)")
            plt.plot(grp["layer"], grp["l2_ft"],   label="FT L2 (orig↔para)")
            plt.xlabel("Layer"); plt.ylabel("Mean L2 distance"); plt.title(f"{where}: orig↔para distances")
            plt.legend(); plt.tight_layout()
            plt.savefig(out_dir / f"{where}_distance_curves.png", dpi=180); plt.close()

            plt.figure()
            plt.plot(grp["layer"], grp["diff_of_diffs_l2"], label="diff-of-diffs = Base − FT")
            plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
            plt.xlabel("Layer"); plt.ylabel("Δ L2 (Base−FT)"); plt.title(f"{where}: diff-of-diffs")
            plt.legend(); plt.tight_layout()
            plt.savefig(out_dir / f"{where}_diff_of_diffs_curve.png", dpi=180); plt.close()

    return df

# =========================
# Weights & PCA & Plots
# =========================
def collect_weight_deltas(base_model: PreTrainedModel, ft_eff_model: PreTrainedModel):
    """
    Compare base vs EFFECTIVE FT model (merged if LoRA). Both should expose same param names/shapes.
    All comparisons are done on CPU to avoid device mismatches.
    """
    rows = []
    with torch.no_grad():
        base_params = dict(base_model.named_parameters())
        ft_params   = dict(ft_eff_model.named_parameters())
        intersect = [n for n in base_params.keys() if n in ft_params and base_params[n].shape == ft_params[n].shape]
        for n_b in intersect:
            p_b = base_params[n_b].detach().to("cpu").float()
            p_f = ft_params[n_b].detach().to("cpu").float()
            scope, bucket, layer_idx = module_bucket(n_b)
            if bucket not in {"q","k","v","o","up","gate","down"}:
                continue
            dW = (p_f - p_b)
            fro = torch.linalg.matrix_norm(dW).item()
            rows.append({
                "name": n_b, "layer": layer_idx, "scope": scope, "bucket": bucket,
                "delta_fro": fro, "shape": str(tuple(p_b.shape))
            })
    return pd.DataFrame(rows)

def choose_culprit_layers(df_dod: pd.DataFrame, where: str, top_k: int, user_layers: Optional[List[int]]) -> List[int]:
    if user_layers:
        return sorted(set(int(x) for x in user_layers))
    sub = df_dod[df_dod["where"] == where]
    if sub.empty:
        return []
    grp = sub.groupby("layer")["diff_of_diffs_l2"].mean().reset_index()
    grp = grp.sort_values("diff_of_diffs_l2", ascending=False)
    return grp.head(top_k)["layer"].astype(int).tolist()

def pca_of_matrix(W: torch.Tensor):
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    S = S.cpu().numpy()
    expl = (S**2) / max((S**2).sum(), 1e-12)
    cum = np.cumsum(expl)
    return S, expl, cum

def plot_spectra_and_cum(S_base, S_ft, expl_base, expl_ft, cum_base, cum_ft, outdir: Path, tag: str):
    plt.figure()
    plt.plot(np.arange(len(S_base)), S_base, label="Base σ")
    plt.plot(np.arange(len(S_ft)),   S_ft,   label="FT σ")
    plt.xlabel("Index"); plt.ylabel("Singular value"); plt.title(f"Spectrum: {tag}")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"spectrum_{tag}.png", dpi=180); plt.close()

    plt.figure()
    plt.plot(np.arange(len(cum_base)), cum_base, label="Base cumulative variance")
    plt.plot(np.arange(len(cum_ft)),   cum_ft,   label="FT cumulative variance")
    plt.xlabel("PCs"); plt.ylabel("Explained variance (cumulative)"); plt.title(f"PCA cumulative: {tag}")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"pca_cumulative_{tag}.png", dpi=180); plt.close()

def run_pca_for_layers(base_model, ft_eff_model, layers: List[int], output_dir: Path):
    outdir = output_dir / "pca_weights"
    outdir.mkdir(parents=True, exist_ok=True)

    base_params = dict(base_model.named_parameters())
    ft_params   = dict(ft_eff_model.named_parameters())

    for layer in layers:
        for bucket in ["o", "down"]:
            tgt_name = None
            # Find parameter name in base (assumed also present in FT-merged)
            for n in base_params.keys():
                if f"layers.{layer}." in n and (n.endswith(".o_proj.weight") if bucket=="o" else n.endswith(".down_proj.weight")):
                    tgt_name = n; break
            if tgt_name is None:
                # Fallback: scan FT params (some models name slightly differently)
                for n in ft_params.keys():
                    if f"layers.{layer}." in n and (n.endswith(".o_proj.weight") if bucket=="o" else n.endswith(".down_proj.weight")):
                        tgt_name = n; break
            if tgt_name is None:
                continue

            Wb = base_params[tgt_name].detach().cpu()
            Wf = ft_params[tgt_name].detach().cpu()
            S_b, expl_b, cum_b = pca_of_matrix(Wb)
            S_f, expl_f, cum_f = pca_of_matrix(Wf)
            tag = f"layer{layer}_{bucket}"
            plot_spectra_and_cum(S_b, S_f, expl_b, expl_f, cum_b, cum_f, outdir, tag)
            with open(outdir / f"pca_stats_{tag}.txt", "w") as fh:
                def ksum(cum, k):
                    k = min(k, len(cum)); return cum[k-1] if k>0 else 0.0
                fh.writelines([
                    f"{tgt_name}\n",
                    f"Top-10 EV (Base): {ksum(cum_b,10):.4f} | Top-20: {ksum(cum_b,20):.4f} | Top-64: {ksum(cum_b,64):.4f}\n",
                    f"Top-10 EV (FT):   {ksum(cum_f,10):.4f} | Top-20: {ksum(cum_f,20):.4f} | Top-64: {ksum(cum_f,64):.4f}\n",
                ])

def plot_top_collapse_bar(df_dod: pd.DataFrame, where: str, outdir: Path, top_layers: List[int]):
    sub = df_dod[df_dod["where"] == where]
    if sub.empty:
        return
    grp = sub.groupby("layer")["diff_of_diffs_l2"].mean().reset_index()
    grp = grp.sort_values("diff_of_diffs_l2", ascending=False)
    plt.figure()
    plt.bar(grp["layer"].astype(int), grp["diff_of_diffs_l2"])
    plt.xlabel("Layer")
    plt.ylabel("Mean diff-of-diffs (Base−FT)")
    plt.title(f"Top collapse layers at {where}")
    plt.tight_layout()
    plt.savefig(outdir / "top_collapse_layers_bar.png", dpi=180); plt.close()

    # Also highlight chosen layers
    if top_layers:
        sel = grp[grp["layer"].isin(top_layers)]
        if not sel.empty:
            plt.figure()
            plt.bar(sel["layer"].astype(int), sel["diff_of_diffs_l2"])
            plt.xlabel("Chosen Layers")
            plt.ylabel("Mean diff-of-diffs (Base−FT)")
            plt.title(f"Chosen layers at {where}")
            plt.tight_layout()
            plt.savefig(outdir / "chosen_layers_diff_of_diffs_bar.png", dpi=180); plt.close()

def plot_weight_delta_heatmap(weights_df: pd.DataFrame, outdir: Path, layers_of_interest: Optional[List[int]]):
    if weights_df.empty:
        return
    piv = weights_df.pivot_table(index="layer", columns="bucket", values="delta_fro", aggfunc="sum").fillna(0.0)
    piv = piv.sort_index()
    plt.figure()
    plt.imshow(piv.values, aspect="auto")
    plt.xticks(ticks=np.arange(piv.shape[1]), labels=list(piv.columns), rotation=45, ha="right")
    plt.yticks(ticks=np.arange(piv.shape[0]), labels=[int(x) for x in piv.index])
    plt.colorbar(label="ΔW Frobenius")
    plt.title("Weight deltas by layer × bucket (effective FT)")
    plt.tight_layout()
    plt.savefig(outdir / "weight_delta_heatmap.png", dpi=180); plt.close()

    # Bars for chosen layers
    if layers_of_interest:
        sub = weights_df[weights_df["layer"].isin(layers_of_interest)]
        if not sub.empty:
            agg = sub.groupby("bucket")["delta_fro"].sum().reindex(["q","k","v","o","up","gate","down"], fill_value=0.0)
            plt.figure()
            plt.bar(agg.index, agg.values)
            plt.xlabel("Bucket"); plt.ylabel("Σ ΔW Fro")
            plt.title(f"Weight deltas — chosen layers {layers_of_interest}")
            plt.tight_layout()
            plt.savefig(outdir / "weight_delta_bars_toplayers.png", dpi=180); plt.close()

# =========================
# CLI
# =========================
def parse_layers_list(txt: Optional[str]) -> Optional[List[int]]:
    if not txt:
        return None
    vals = []
    for part in txt.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part))
        except ValueError:
            pass
    return vals if vals else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_path", required=True)
    ap.add_argument("--ft_model_path", required=True)
    ap.add_argument("--prompts_json_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16","float32"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pairs_per_batch", type=int, default=8)
    ap.add_argument("--focus_on_answer_tokens", action="store_true")
    ap.add_argument("--run_mode", default="extra_suite", choices=["resid_dod","extra_suite"])
    ap.add_argument("--where_collapse", default="resid_post", choices=["resid_pre","resid_mid","resid_post"],
                    help="Which residual site to rank for top collapse layers.")
    ap.add_argument("--top_k_culprit", type=int, default=3, help="How many top layers to analyze if no culprit_layers given.")
    ap.add_argument("--culprit_layers", type=str, default="",
                    help="Comma-separated explicit layer indices to analyze instead of top-k (e.g., '6,9,15').")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("CUDA not available; this will be slow.")

    # Load models/tokenizer
    base_model, tokenizer, base_info = load_model_and_tokenizer(args.base_model_path, args.base_model_path, device, args.dtype)
    ft_model, _tok2, ft_info = load_model_and_tokenizer(args.ft_model_path, args.base_model_path, device, args.dtype)

    prompts = load_prompts(args.prompts_json_path, args.limit)
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    # A) Diff-of-diffs at resid_pre/mid/post (forward pass uses PEFT if provided)
    df_dod = run_resid_dod(
        base_model, ft_model, tokenizer, prompts, device, outdir,
        focus_on_answer=args.focus_on_answer_tokens,
        pairs_per_batch=args.pairs_per_batch, limit=args.limit
    )

    if args.run_mode == "resid_dod":
        log.info("Saved residual diff-of-diffs outputs.")
        if not df_dod.empty:
            plot_top_collapse_bar(df_dod, args.where_collapse, outdir / "resid_dod", top_layers=[])
        return

    # B) Build EFFECTIVE FT model for weight deltas/PCA (merge LoRA on CPU if needed)
    ft_eff = build_effective_ft_model_for_weights(ft_info)
    if ft_eff is None:
        # Not PEFT (or merge failed): use the FT model directly
        ft_eff = ft_model

    # Collect ΔW (effective) and plot
    weights_df = collect_weight_deltas(base_model, ft_eff)
    weights_df.to_csv(outdir / "weight_deltas_by_bucket.csv", index=False)

    # C) Pick culprit layers (either user-provided or top-k) and run PCA
    culprit_layers = choose_culprit_layers(df_dod, where=args.where_collapse,
                                           top_k=args.top_k_culprit,
                                           user_layers=parse_layers_list(args.culprit_layers))
    pd.DataFrame({"layer": culprit_layers}).to_csv(outdir / "top_collapse_layers.csv", index=False)

    # Plots associated with these CSVs
    if not df_dod.empty:
        plot_top_collapse_bar(df_dod, args.where_collapse, outdir, top_layers=culprit_layers)
    plot_weight_delta_heatmap(weights_df, outdir, layers_of_interest=culprit_layers)

    # PCA on o_proj/down_proj for the culprit layers (effective FT vs base)
    run_pca_for_layers(base_model, ft_eff, culprit_layers, outdir)

    log.info("extra suite complete. Outputs saved under %s", outdir)

if __name__ == "__main__":
    main()
