import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

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

# Logging
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

# Helpers
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

# ---------- model/layer introspection ----------
def _maybe_get_base_model(model: nn.Module) -> nn.Module:
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
    m = _maybe_get_base_model(model)
    if hasattr(m, "model") and hasattr(m.model, "layers") and isinstance(m.model.layers, nn.ModuleList):
        return list(m.model.layers)
    for _, mod in m.named_modules():
        if hasattr(mod, "layers") and isinstance(mod.layers, nn.ModuleList):
            return list(mod.layers)
    candidates = []
    for _, mod in m.named_modules():
        if hasattr(mod, "self_attn") and hasattr(mod, "mlp"):
            candidates.append(mod)
    if candidates:
        return candidates
    raise TypeError("Could not locate decoder layers for hooks; model structure not recognized.")

# Tokenisation with segment masks
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

# Probes
class LayerIOProbe:
    """
    Captures per-layer:
      - layer_input (resid_pre)
      - attn_o (self_attn.o_proj output)
      - mlp_down (mlp.down_proj output)
      - layer_output (resid_post)
    resid_mid is reconstructed as layer_input + attn_o.

    Works for bare and PEFT-wrapped models.
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

# Data & Models
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
    try:
        base_cpu = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16, device_map="cpu")
        ft_cpu = PeftModel.from_pretrained(base_cpu, adapter_path, torch_dtype=torch.float16, device_map="cpu")
        ft_cpu = ft_cpu.merge_and_unload()
        ft_cpu.eval()
        log.info("Built effective merged FT model on CPU for weight diffs/PCA and ablations.")
        return ft_cpu
    except Exception as e:
        log.warning("Failed to build merged FT model on CPU: %s", e)
        return None

# Core: diff-of-diffs on residual stream
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
                  pairs_per_batch: int = 8, limit: Optional[int] = None,
                  tag_prefix: str = ""):
    """
    Compute diff-of-diffs at three residual sites:
      - resid_pre  (layer input, before attention and MLP)
      - resid_mid  (after attention output is added)
      - resid_post (layer output, after MLP)
    For each layer/site, measure L2(orig↔para) on Base vs FT (or Patched), then (Base − FT).

    If tag_prefix is provided, CSV/plots are named with that prefix to separate ablations.
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
    tag_mid = (tag_prefix + "_") if tag_prefix else ""
    df.to_csv(out_dir / f"{tag_mid}resid_diff_of_diffs_by_layer.csv", index=False)

    if not df.empty:
        for where in ["resid_pre","resid_mid","resid_post"]:
            sub = df[df["where"] == where]
            grp = sub.groupby("layer")[["l2_base","l2_ft","diff_of_diffs_l2"]].mean().reset_index()

            plt.figure()
            plt.plot(grp["layer"], grp["l2_base"], label="Base L2 (orig↔para)")
            plt.plot(grp["layer"], grp["l2_ft"],   label="FT L2 (orig↔para)")
            plt.xlabel("Layer"); plt.ylabel("Mean L2 distance"); plt.title(f"{tag_prefix} {where}: orig↔para distances".strip())
            plt.legend(); plt.tight_layout()
            plt.savefig(out_dir / f"{tag_mid}{where}_distance_curves.png", dpi=180); plt.close()

            plt.figure()
            plt.plot(grp["layer"], grp["diff_of_diffs_l2"], label="diff-of-diffs = Base − FT")
            plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
            plt.xlabel("Layer"); plt.ylabel("Δ L2 (Base−FT)"); plt.title(f"{tag_prefix} {where}: diff-of-diffs".strip())
            plt.legend(); plt.tight_layout()
            plt.savefig(out_dir / f"{tag_mid}{where}_diff_of_diffs_curve.png", dpi=180); plt.close()

    return df

# Weights & PCA & Plots
def collect_weight_deltas(base_model: PreTrainedModel, ft_eff_model: PreTrainedModel):
    """
    Compare base vs EFFECTIVE FT model (merged if LoRA). Do all diffs on CPU to avoid device mismatches.
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
            for n in base_params.keys():
                if f"layers.{layer}." in n and (n.endswith(".o_proj.weight") if bucket=="o" else n.endswith(".down_proj.weight")):
                    tgt_name = n; break
            if tgt_name is None:
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

# Residual-site contributions (attn vs mlp)
def compute_residual_contributions(df_dod: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rdir = outdir / "resid_dod"
    rdir.mkdir(parents=True, exist_ok=True)
    if df_dod.empty:
        return df_dod

    pivot = df_dod.pivot_table(index="layer", columns="where", values="diff_of_diffs_l2", aggfunc="mean")
    for k in ["resid_pre","resid_mid","resid_post"]:
        if k not in pivot.columns:
            pivot[k] = np.nan
    pivot = pivot.sort_index()
    pivot["Delta_attn"] = pivot["resid_mid"] - pivot["resid_pre"]
    pivot["Delta_mlp"]  = pivot["resid_post"] - pivot["resid_mid"]
    pivot.rename(columns={
        "resid_pre": "Delta_pre",
        "resid_mid": "Delta_mid",
        "resid_post": "Delta_post"
    }, inplace=True)
    pivot.to_csv(rdir / "residual_contrib_by_layer.csv")

    plt.figure()
    plt.plot(pivot.index, pivot["Delta_attn"], label="Δ_attn (mid − pre)")
    plt.plot(pivot.index, pivot["Delta_mlp"],  label="Δ_mlp (post − mid)")
    plt.axhline(0.0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Layer"); plt.ylabel("Contribution to diff-of-diffs")
    plt.title("Per-layer contributions: attention vs MLP")
    plt.legend(); plt.tight_layout()
    plt.savefig(rdir / "residual_contrib_lines.png", dpi=180); plt.close()

    plt.figure()
    width = 0.25
    x = np.arange(len(pivot.index))
    plt.bar(x - width, pivot["Delta_pre"],  width=width, label="Δ_pre")
    plt.bar(x,         pivot["Delta_mid"],  width=width, label="Δ_mid")
    plt.bar(x + width, pivot["Delta_post"], width=width, label="Δ_post")
    plt.xlabel("Layer"); plt.ylabel("Mean diff-of-diffs")
    plt.title("Mean diff-of-diffs at residual sites")
    plt.legend(); plt.tight_layout()
    plt.savefig(rdir / "residual_sites_bars.png", dpi=180); plt.close()

    return pivot

# Activation PCA / Effective Rank (optional, unchanged from your last version)
def participation_ratio(eigs: np.ndarray) -> float:
    s1 = float(eigs.sum())
    s2 = float((eigs**2).sum())
    if s2 <= 0:
        return 0.0
    return (s1 * s1) / s2

def entropy_effective_rank(eigs: np.ndarray, eps: float = 1e-12) -> float:
    p = eigs / max(eigs.sum(), eps)
    p = np.clip(p, eps, 1.0)
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))

def _extract_site_tensor(pre: torch.Tensor, attno: torch.Tensor, post: torch.Tensor, site: str) -> torch.Tensor:
    if site == "resid_pre":  return pre
    if site == "resid_mid":  return pre + attno
    if site == "resid_post": return post
    raise ValueError(f"Unknown site {site}")

def _append_activations(storage: Dict[Tuple[int,str], List[np.ndarray]],
                        layers: Iterable[int],
                        sites: Iterable[str],
                        probe: LayerIOProbe,
                        masks: torch.Tensor,
                        pooling: str = "mean",
                        token_stride: int = 4,
                        max_tokens_per_item: Optional[int] = None):
    B, S = masks.shape
    for l in layers:
        pre  = probe.layer_input.get(l)
        post = probe.layer_output.get(l)
        attn = probe.attn_o.get(l)
        if pre is None or post is None or attn is None:
            continue
        for i in range(B):
            m = masks[i]
            if not torch.any(m):
                continue
            for site in sites:
                t = _extract_site_tensor(pre, attn, post, site)[i]  # [S,H]
                if pooling == "mean":
                    vec = masked_mean(t, m).detach().cpu().float().numpy()
                    storage[(l, site)].append(vec)
                else:
                    idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
                    if token_stride > 1:
                        idx = idx[::token_stride]
                    if max_tokens_per_item is not None and len(idx) > max_tokens_per_item:
                        idx = idx[:max_tokens_per_item]
                    sample = t.index_select(0, idx).detach().cpu().float().numpy()  # [T,H]
                    for row in sample:
                        storage[(l, site)].append(row)

def activation_pca_from_matrix(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if X.ndim != 2 or X.shape[0] < 2:
        return np.array([]), np.array([]), np.array([]), 0.0, 0.0
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigs = (S ** 2)
    ev = eigs / max(eigs.sum(), 1e-12)
    cum = np.cumsum(ev)
    PR = participation_ratio(eigs)
    eR = entropy_effective_rank(eigs)
    return S, ev, cum, PR, eR

def run_activation_pca(base_model, ft_model, tokenizer, prompts: List[dict],
                       layers: List[int],
                       sites: List[str],
                       device: str,
                       output_dir: Path,
                       pairs_per_batch: int = 8,
                       limit: Optional[int] = None,
                       pooling: str = "mean",
                       token_stride: int = 4,
                       max_tokens_per_item: Optional[int] = None,
                       max_items_for_pca: Optional[int] = None):
    outdir = output_dir / "activation_pca"
    outdir.mkdir(parents=True, exist_ok=True)

    def paraphrase_keys(d):
        return [k for k in d.keys() if k.startswith("instruct_") and k != "instruction_original"]
    selected = prompts if limit is None else prompts[:limit]

    base_store: Dict[Tuple[int,str], List[np.ndarray]] = { (l,s): [] for l in layers for s in sites }
    ft_store:   Dict[Tuple[int,str], List[np.ndarray]] = { (l,s): [] for l in layers for s in sites }

    n_items_seen = 0
    for batch_start in range(0, len(selected), pairs_per_batch):
        if max_items_for_pca is not None and n_items_seen >= max_items_for_pca:
            break
        batch = selected[batch_start: batch_start + pairs_per_batch]

        encs = []
        for ex in batch:
            instr = ex.get("instruction_original","")
            inp   = ex.get("input","") or ""
            keys = paraphrase_keys(ex)
            if not keys:
                continue
            pkey = keys[0]
            para = ex.get(pkey, "")
            encs.append(tokenise_with_segments(tokenizer, instr, inp, device, answer_text=None))
            encs.append(tokenise_with_segments(tokenizer, para,  inp, device, answer_text=None))

        if not encs:
            continue

        pad_side = getattr(tokenizer, "padding_side", "left")
        batch_enc = pad_and_stack(encs, tokenizer.pad_token_id, padding_side=pad_side)
        masks = batch_enc["attention_mask"].bool()

        with LayerIOProbe(base_model) as probe_b:
            _ = base_model(batch_enc["input_ids"], attention_mask=batch_enc["attention_mask"], use_cache=True)
        _append_activations(base_store, layers, sites, probe_b, masks,
                            pooling=pooling, token_stride=token_stride, max_tokens_per_item=max_tokens_per_item)

        with LayerIOProbe(ft_model) as probe_f:
            _ = ft_model(batch_enc["input_ids"], attention_mask=batch_enc["attention_mask"], use_cache=True)
        _append_activations(ft_store, layers, sites, probe_f, masks,
                            pooling=pooling, token_stride=token_stride, max_tokens_per_item=max_tokens_per_item)

        n_items_seen += (len(encs))
        if max_items_for_pca is not None and n_items_seen >= max_items_for_pca:
            break

    summary_rows = []
    for l in layers:
        for site in sites:
            Xb = np.stack(base_store[(l,site)], axis=0) if base_store[(l,site)] else np.zeros((0,1))
            Xf = np.stack(ft_store[(l,site)],   axis=0) if ft_store[(l,site)] else np.zeros((0,1))

            Sb, evb, cmb, PRb, eRb = activation_pca_from_matrix(Xb)
            Sf, evf, cmf, PRf, eRf = activation_pca_from_matrix(Xf)

            tag = f"layer{l}_{site}"
            plt.figure()
            plt.plot(np.arange(len(Sb)), Sb, label="Base σ")
            plt.plot(np.arange(len(Sf)), Sf, label="FT σ")
            plt.xlabel("Index"); plt.ylabel("Singular value"); plt.title(f"Activation spectrum: {tag}")
            plt.legend(); plt.tight_layout()
            plt.savefig(outdir / f"activation_spectrum_{tag}.png", dpi=180); plt.close()

            plt.figure()
            plt.plot(np.arange(len(cmb)), cmb, label="Base cumulative variance")
            plt.plot(np.arange(len(cmf)), cmf, label="FT cumulative variance")
            plt.xlabel("PCs"); plt.ylabel("Explained variance (cumulative)"); plt.title(f"PCA cumulative (activations): {tag}")
            plt.legend(); plt.tight_layout()
            plt.savefig(outdir / f"activation_cumulative_{tag}.png", dpi=180); plt.close()

            def topk(cum: np.ndarray, k: int) -> float:
                if len(cum) == 0: return 0.0
                k = min(k, len(cum)); 
                return float(cum[k-1]) if k > 0 else 0.0

            with open(outdir / f"activation_pca_stats_{tag}.txt", "w") as fh:
                fh.writelines([
                    f"{tag}\n",
                    f"Rows (Base/FT): {Xb.shape[0]}/{Xf.shape[0]} | H: {Xb.shape[1] if Xb.size>0 else 'NA'}\n",
                    f"Top-10 EV (Base/FT): {topk(cmb,10):.4f} / {topk(cmf,10):.4f}\n",
                    f"Top-20 EV (Base/FT): {topk(cmb,20):.4f} / {topk(cmf,20):.4f}\n",
                    f"Top-64 EV (Base/FT): {topk(cmb,64):.4f} / {topk(cmf,64):.4f}\n",
                    f"Participation ratio (Base/FT): {PRb:.2f} / {PRf:.2f}\n",
                    f"Entropy effective rank (Base/FT): {eRb:.2f} / {eRf:.2f}\n",
                ])

            summary_rows.append({
                "layer": l, "site": site,
                "rows_base": int(Xb.shape[0]), "rows_ft": int(Xf.shape[0]),
                "pr_base": PRb, "pr_ft": PRf,
                "erank_base": eRb, "erank_ft": eRf,
                "top10_base": topk(cmb,10), "top10_ft": topk(cmf,10),
                "top20_base": topk(cmb,20), "top20_ft": topk(cmf,20),
                "top64_base": topk(cmb,64), "top64_ft": topk(cmf,64),
            })

    pd.DataFrame(summary_rows).to_csv(outdir / "activation_pca_summary.csv", index=False)

# Layer-6 ablation suite (heavy-lift test)
def list_param_names_for(layer: int, buckets: Iterable[str], name_list: List[str]) -> List[str]:
    out = []
    for n in name_list:
        if f"layers.{layer}." not in n:
            continue
        for b in buckets:
            if b == "q" and n.endswith(".self_attn.q_proj.weight"): out.append(n)
            if b == "k" and n.endswith(".self_attn.k_proj.weight"): out.append(n)
            if b == "v" and n.endswith(".self_attn.v_proj.weight"): out.append(n)
            if b == "o" and n.endswith(".self_attn.o_proj.weight"): out.append(n)
            if b == "up" and n.endswith(".mlp.up_proj.weight"): out.append(n)
            if b == "gate" and n.endswith(".mlp.gate_proj.weight"): out.append(n)
            if b == "down" and n.endswith(".mlp.down_proj.weight"): out.append(n)
    return sorted(set(out))

def deepcopy_model(model: nn.Module) -> nn.Module:
    # torch.clone doesn't clone modules; we use Python deepcopy
    import copy
    return copy.deepcopy(model)

def patch_from_source(dst_model: nn.Module, src_params: Dict[str, torch.Tensor], names: List[str]):
    with torch.no_grad():
        dmap = dict(dst_model.named_parameters())
        for n in names:
            if n not in dmap or n not in src_params:
                continue
            p = dmap[n]
            p.data.copy_(src_params[n].to(p.device, dtype=p.dtype))

def run_ablation_suite(base_model, ft_eff_model, tokenizer, prompts, device, outdir: Path,
                       layer: int, ablation_kinds: List[str],
                       pairs_per_batch: int, limit: Optional[int],
                       where_for_fraction: str,
                       full_df_dod: pd.DataFrame):
    """
    For each ablation kind we create a copy of the base model, patch only the requested layer-6 weights
    from the EFFECTIVE FT model, and compute Base vs Patched diff-of-diffs. We also compute
    'fraction of full' = (Base−Patched) / (Base−FT).
    """
    out_ab = outdir / "ablation_suite"
    out_ab.mkdir(parents=True, exist_ok=True)

    ft_params_cpu = {n: p.detach().cpu() for n, p in ft_eff_model.named_parameters()}
    base_param_names = [n for n, _ in base_model.named_parameters()]

    # Define bucket groups
    presets = {
        "attn_all":     ["q","k","v","o"],
        "attn_qkv":     ["q","k","v"],
        "attn_o":       ["o"],
        "mlp_all":      ["up","gate","down"],
        "mlp_up_gate":  ["up","gate"],
        "mlp_down":     ["down"],
    }

    summary_rows = []

    for kind in ablation_kinds:
        if kind not in presets:
            log.warning("Unknown ablation kind '%s' (skipping).", kind)
            continue
        buckets = presets[kind]
        names = list_param_names_for(layer, buckets, base_param_names)
        if not names:
            log.warning("No parameter names matched for %s at layer %d", kind, layer)
            continue

        # Create patched model = deepcopy(base) + selected params from ft_eff
        patched = deepcopy_model(base_model)
        patch_from_source(patched, ft_params_cpu, names)

        # Run Base vs Patched diff-of-diffs
        df_patch = run_resid_dod(
            base_model, patched, tokenizer, prompts, device, outdir,
            pairs_per_batch=pairs_per_batch, limit=limit, tag_prefix=f"ablate_{kind}"
        )

        # Compare to full diff-of-diffs (Base vs full FT)
        # 'fraction of full' per (layer, where)
        if not df_patch.empty and not full_df_dod.empty:
            agg_patch = df_patch.groupby(["layer","where"])["diff_of_diffs_l2"].mean().reset_index()
            agg_full  = full_df_dod.groupby(["layer","where"])["diff_of_diffs_l2"].mean().reset_index()
            joined = pd.merge(agg_patch, agg_full, on=["layer","where"], suffixes=("_patch","_full"))
            joined["fraction_of_full"] = np.where(
                np.abs(joined["diff_of_diffs_l2_full"]) > 1e-9,
                joined["diff_of_diffs_l2_patch"] / joined["diff_of_diffs_l2_full"],
                np.nan
            )
            joined.to_csv(out_ab / f"ablation_fraction_of_full_{kind}.csv", index=False)

            # Keep a summary row per site
            for site in ["resid_pre","resid_mid","resid_post"]:
                sub = joined[joined["where"] == site]
                frac_mean = float(sub["fraction_of_full"].dropna().mean()) if not sub.empty else float("nan")
                summary_rows.append({"kind": kind, "site": site, "mean_fraction_of_full": frac_mean})

            # Plot heatmap-style line per site
            for site in ["resid_pre","resid_mid","resid_post"]:
                sub = joined[joined["where"] == site].sort_values("layer")
                if sub.empty: 
                    continue
                plt.figure()
                plt.plot(sub["layer"].astype(int), sub["fraction_of_full"])
                plt.xlabel("Layer"); plt.ylabel("Fraction of full diff-of-diffs")
                plt.title(f"Ablation {kind}: fraction of full at {site}")
                plt.axhline(1.0, linestyle="--", color="k", linewidth=1)
                plt.tight_layout()
                plt.savefig(out_ab / f"ablation_fraction_of_full_{kind}_{site}.png", dpi=180); plt.close()

        # Free
        del patched
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_piv = summary.pivot_table(index="kind", columns="site", values="mean_fraction_of_full", aggfunc="mean")
        summary_piv.to_csv(out_ab / "ablation_summary_fraction_of_full.csv")

        # Site-specific bars
        for site in ["resid_pre","resid_mid","resid_post"]:
            if site not in summary_piv.columns: 
                continue
            plt.figure()
            vals = summary_piv[site].fillna(0.0)
            plt.bar(vals.index, vals.values)
            plt.ylabel("Mean fraction of full")
            plt.title(f"Ablation summary at {site}")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(out_ab / f"ablation_summary_fraction_of_full_{site}.png", dpi=180); plt.close()

# CLI utilities
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

def parse_sites_list(txt: Optional[str]) -> List[str]:
    if not txt:
        return ["resid_mid", "resid_post"]
    out = []
    for p in txt.split(","):
        p = p.strip()
        if p in {"resid_pre","resid_mid","resid_post"}:
            out.append(p)
    return out or ["resid_mid","resid_post"]

# Main
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
    ap.add_argument("--run_mode", default="extra_suite",
                    choices=["resid_dod","extra_suite","activation_only"])
    ap.add_argument("--where_collapse", default="resid_post", choices=["resid_pre","resid_mid","resid_post"],
                    help="Which residual site to rank for top collapse layers.")
    ap.add_argument("--top_k_culprit", type=int, default=3,
                    help="How many top layers to analyze if no --culprit_layers given.")
    ap.add_argument("--culprit_layers", type=str, default="",
                    help="Comma-separated layer indices to analyze instead of top-k (e.g., '6,23,24').")

    # Activation PCA options
    ap.add_argument("--activation_layers", type=str, default="",
                    help="Comma-separated layers for activation PCA. If empty, falls back to culprit layers.")
    ap.add_argument("--activation_sites", type=str, default="resid_mid,resid_post",
                    help="Comma-separated sites among {resid_pre,resid_mid,resid_post}.")
    ap.add_argument("--activation_pool", type=str, default="mean", choices=["mean","tokens"])
    ap.add_argument("--token_stride", type=int, default=4)
    ap.add_argument("--max_tokens_per_item", type=int, default=None)
    ap.add_argument("--max_items_for_pca", type=int, default=None)

    # Ablation suite
    ap.add_argument("--run_ablation_suite", action="store_true",
                    help="If set, run layer-level ablations to find heavy-lifting weights.")
    ap.add_argument("--ablate_layer", type=int, default=6,
                    help="Which layer to ablate (copy weights from FT into base).")
    ap.add_argument("--ablation_kinds", type=str,
                    default="attn_all,attn_qkv,attn_o,mlp_all,mlp_up_gate,mlp_down",
                    help="Comma-separated ablation presets among {attn_all,attn_qkv,attn_o,mlp_all,mlp_up_gate,mlp_down}.")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("CUDA not available; this will be slow.")

    base_model, tokenizer, base_info = load_model_and_tokenizer(args.base_model_path, args.base_model_path, device, args.dtype)
    ft_model, _tok2, ft_info = load_model_and_tokenizer(args.ft_model_path, args.base_model_path, device, args.dtype)

    prompts = load_prompts(args.prompts_json_path, args.limit)
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)

    # A) Residual diff-of-diffs (Base vs full FT)
    df_dod_full = run_resid_dod(
        base_model, ft_model, tokenizer, prompts, device, outdir,
        focus_on_answer=args.focus_on_answer_tokens,
        pairs_per_batch=args.pairs_per_batch, limit=args.limit
    )
    compute_residual_contributions(df_dod_full, outdir)

    if args.run_mode == "resid_dod":
        log.info("Saved residual diff-of-diffs outputs + attribution.")
        return

    # B) Effective FT model for weight diffs/PCA and ablations
    ft_eff = build_effective_ft_model_for_weights(ft_info)
    if ft_eff is None:
        ft_eff = ft_model

    weights_df = collect_weight_deltas(base_model, ft_eff)
    weights_df.to_csv(outdir / "weight_deltas_by_bucket.csv", index=False)

    culprit_layers = choose_culprit_layers(df_dod_full, where=args.where_collapse,
                                           top_k=args.top_k_culprit,
                                           user_layers=parse_layers_list(args.culprit_layers))
    pd.DataFrame({"layer": culprit_layers}).to_csv(outdir / "top_collapse_layers.csv", index=False)

    plot_top_collapse_bar(df_dod_full, args.where_collapse, outdir, top_layers=culprit_layers)
    plot_weight_delta_heatmap(weights_df, outdir, layers_of_interest=culprit_layers)
    run_pca_for_layers(base_model, ft_eff, culprit_layers, outdir)

    # C) Activation PCA (optional)
    act_layers = parse_layers_list(args.activation_layers) or culprit_layers
    act_sites  = parse_sites_list(args.activation_sites)
    if act_layers:
        run_activation_pca(
            base_model, ft_model, tokenizer, prompts,
            layers=act_layers,
            sites=act_sites,
            device=device,
            output_dir=outdir,
            pairs_per_batch=args.pairs_per_batch,
            limit=args.limit,
            pooling=args.activation_pool,
            token_stride=args.token_stride,
            max_tokens_per_item=args.max_tokens_per_item,
            max_items_for_pca=args.max_items_for_pca
        )

    # D) Ablation suite (heavy-lift test)
    if args.run_ablation_suite:
        kinds = [k.strip() for k in args.ablation_kinds.split(",") if k.strip()]
        run_ablation_suite(
            base_model, ft_eff, tokenizer, prompts, device, outdir,
            layer=args.ablate_layer, ablation_kinds=kinds,
            pairs_per_batch=args.pairs_per_batch, limit=args.limit,
            where_for_fraction=args.where_collapse,
            full_df_dod=df_dod_full
        )

    log.info("Suite complete. Outputs saved under %s", outdir)

if __name__ == "__main__":
    main()
