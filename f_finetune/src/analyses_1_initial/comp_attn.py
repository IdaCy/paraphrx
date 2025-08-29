# expert_attention_tokens.py
from __future__ import annotations

import argparse, json, re, math, os, time
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

from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# ------------------------------
# Utils & data
# ------------------------------

def set_seed(seed=42):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

@dataclass
class Item:
    prompt_count: int
    instruction_original: str
    paraphrases: Dict[str, str]
    input_text: str

    def get_paraphrase_keys(self, keys: Optional[List[str]], regex: Optional[str]) -> List[str]:
        all_keys = [k for k in sorted(self.paraphrases.keys()) if k.startswith("instruct_")]
        if keys:
            keep = [k for k in all_keys if k in keys]
        elif regex:
            rgx = re.compile(regex)
            keep = [k for k in all_keys if rgx.search(k)]
        else:
            keep = all_keys
        return keep

def load_instruction_json(path: str | Path) -> List[Item]:
    data = json.loads(Path(path).read_text())
    items: List[Item] = []
    for obj in data:
        items.append(
            Item(
                prompt_count=int(obj["prompt_count"]),
                instruction_original=obj.get("instruction_original",""),
                paraphrases={k:v for k,v in obj.items() if k.startswith("instruct_")},
                input_text=(obj.get("input","") or "")
            )
        )
    return items

def build_prompt_text(instruction: str, input_text: str) -> str:
    return f"{instruction}\n\nInput: {input_text}" if (input_text and input_text.strip()) else instruction


# ------------------------------
# Model & hooks (robust loader)
# ------------------------------

class BlockAccessor:
    def __init__(self, model: nn.Module, layer_index: int):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layers = model.transformer.h
        else:
            raise TypeError("Unsupported model architecture.")
        self.block = layers[layer_index]
        self.attn = getattr(self.block, "self_attn", None) or getattr(self.block, "attention", None)
        if self.attn is None:
            raise TypeError("Could not access attention submodule.")

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
    # Example: ".../mfinal" -> ".../final"
    if p.name.startswith("m") and len(p.name) > 1:
        cand = p.with_name(p.name[1:])
        if cand.exists():
            print(f"[resolver] Fixed likely typo: {p} -> {cand}")
            return cand
    # Also try removing stray trailing slashes handled by Path anyway; nothing else to do here
    return None

def _search_parent_for_adapter(p: Path) -> Optional[Path]:
    parent = p.parent if p.parent.exists() else None
    if not parent: return None
    candidates = []
    for child in parent.iterdir():
        if child.is_dir() and _is_adapter_dir(child):
            mtime = child.stat().st_mtime
            candidates.append((child, mtime))
    if not candidates: return None
    # Prefer a dir named 'final', else newest by mtime
    finals = [c for c,_ in candidates if c.name.lower() == "final"]
    if finals:
        print(f"[resolver] Using adapter dir named 'final' under parent: {finals[0]}")
        return finals[0]
    chosen = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
    print(f"[resolver] Using most recent adapter dir under parent: {chosen}")
    return chosen

def _resolve_local_adapter_path(raw: str) -> Optional[Path]:
    p = Path(raw)
    if p.exists():
        return p
    fix = _try_fix_common_typo(p)
    if fix is not None:
        return fix
    found = _search_parent_for_adapter(p)
    if found is not None:
        return found
    return None

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

    # Prefer adapter path if provided
    if ft_lora_adapter is not None:
        local_path = _resolve_local_adapter_path(ft_lora_adapter)
        if local_path is not None:
            # Local path resolved; decide adapter vs merged
            if _is_adapter_dir(local_path):
                if not _HAS_PEFT:
                    raise RuntimeError("peft not installed but --ft_lora_adapter was provided.")
                print(f"[loader] Loading BASE + LoRA adapter from local dir: {local_path}")
                ft = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
                ft = PeftModel.from_pretrained(ft, str(local_path))
                ft = ft.merge_and_unload().eval()
                return base, ft, tokenizer
            if _is_merged_model_dir(local_path):
                print(f"[loader] Detected merged FT model at --ft_lora_adapter path. Using merged FT: {local_path}")
                ft = AutoModelForCausalLM.from_pretrained(str(local_path), torch_dtype=dtype).to(device).eval()
                return base, ft, tokenizer
            # If we got here, the local path exists but doesn't look like adapter or merged:
            raise ValueError(
                f"[loader] Path exists but is neither adapter nor merged model: {local_path}\n"
                f"Expected 'adapter_config.json' OR HF model files (e.g., config.json + model.safetensors)."
            )
        # Not a local path → *maybe* it's a Hub repo id for an adapter. Try once.
        if not _HAS_PEFT:
            raise RuntimeError("peft not installed and --ft_lora_adapter did not resolve to a local directory.")
        print(f"[loader] Trying to load LoRA adapter from Hub repo id: {ft_lora_adapter}")
        ft = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
        ft = PeftModel.from_pretrained(ft, ft_lora_adapter)
        ft = ft.merge_and_unload().eval()
        return base, ft, tokenizer

    # Else: merged FT path
    mpath = Path(ft_model_name_or_path)
    if mpath.exists():
        print(f"[loader] Loading merged FT from local dir: {mpath}")
        ft = AutoModelForCausalLM.from_pretrained(str(mpath), torch_dtype=dtype).to(device).eval()
        return base, ft, tokenizer
    print(f"[loader] Loading merged FT from Hub repo id: {ft_model_name_or_path}")
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
    attn_mask = enc["attention_mask"][0]
    T = input_ids.shape[0]
    prompt_mask = torch.ones(T, dtype=torch.bool)
    if prompt_span == "no_bos" and T > 0:
        prompt_mask[0] = False
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    return Encoded(input_ids=input_ids.to(device),
                   attention_mask=attn_mask.to(device),
                   prompt_mask=prompt_mask.to(device),
                   tokens=tokens)

class HookHandles:
    def __init__(self): self.handles=[]
    def add(self, h): self.handles.append(h)
    def remove_all(self):
        for h in self.handles:
            try: h.remove()
            except Exception: pass
        self.handles.clear()

def capture_attn(model, layer_index: int, device: str):
    block = BlockAccessor(model, layer_index).block
    attn = getattr(block, "self_attn", None) or getattr(block, "attention", None)
    attn_probs_list: List[torch.Tensor] = []
    hooks = HookHandles()
    def attn_forward_hook(module, inputs, outputs):
        if isinstance(outputs, tuple) and len(outputs) >= 2 and outputs[1] is not None:
            attn_probs_list.append(outputs[1].detach().to("cpu"))  # [B,H,T,T]
    hooks.add(attn.register_forward_hook(attn_forward_hook))

    def run(encoded: Encoded) -> torch.Tensor:
        with torch.no_grad():
            _ = model(input_ids=encoded.input_ids.unsqueeze(0).to(device),
                      attention_mask=encoded.attention_mask.unsqueeze(0).to(device),
                      output_attentions=True, return_dict=True)
        if not attn_probs_list:
            raise RuntimeError("No attention captured. Ensure output_attentions=True and correct layer.")
        A = attn_probs_list[-1][0]  # [H,T,T]
        attn_probs_list.clear()
        return A
    run.hooks = hooks
    return run


# ------------------------------
# Attention comparison & selection
# ------------------------------

def flatten_crop_attn(A: torch.Tensor, mask: torch.Tensor, max_tokens: Optional[int]=None) -> torch.Tensor:
    idx = mask.nonzero(as_tuple=True)[0]
    if max_tokens is not None and idx.numel() > max_tokens:
        idx = idx[:max_tokens]
    idx = idx.to(A.device)
    A = A[idx][:, idx]
    return A.reshape(-1)

def diagonality_ratios(A: torch.Tensor) -> Tuple[float,float]:
    T = A.shape[0]
    total = float(A.sum().item()) + 1e-9
    diag_w1 = float(A.diag().sum().item()) / total
    band_mask = torch.zeros_like(A, dtype=torch.bool)
    rng = torch.arange(T)
    band_mask[rng[:,None], rng[None,:]] = True
    if T > 1:
        band_mask[rng[1:,None], rng[:-1][None,:]] = True
        band_mask[rng[:-1,None], rng[1:][None,:]] = True
    diag_w2 = float(A[band_mask].sum().item()) / total
    return diag_w1, diag_w2

def humanize_tokens(tokens: List[str], max_tokens: Optional[int]) -> List[str]:
    toks = tokens[:max_tokens] if (max_tokens and len(tokens)>max_tokens) else tokens
    out = []
    for t in toks:
        s = t.replace("▁", " ").strip()
        if len(s) > 18: s = s[:15] + "…"
        out.append(s if s else "␠")
    return out


# ------------------------------
# Core run
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize FT token-to-token attention (expert request)")
    ap.add_argument("--instructions_json", type=str, required=True)
    ap.add_argument("--base_model_name_or_path", type=str, required=True)
    ap.add_argument("--ft_model_name_or_path", type=str, default=None)
    ap.add_argument("--ft_lora_adapter", type=str, default=None)

    ap.add_argument("--layer_index", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--prompt_count", type=int, required=True)
    ap.add_argument("--n_paraphrases", type=int, default=8)
    ap.add_argument("--include_original", type=str, default="false", choices=["true","false"])
    ap.add_argument("--keys", type=str, default=None)
    ap.add_argument("--keys_regex", type=str, default="instruct_.*")
    ap.add_argument("--max_tokens", type=int, default=96)

    ap.add_argument("--select_heads_topk", type=int, default=6)
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = ensure_dir(args.outdir)

    items = load_instruction_json(args.instructions_json)
    item = next((it for it in items if it.prompt_count == args.prompt_count), None)
    if item is None:
        raise ValueError(f"prompt_count={args.prompt_count} not found in {args.instructions_json}")

    explicit_keys = [k.strip() for k in args.keys.split(",")] if args.keys else None
    cand_keys = item.get_paraphrase_keys(explicit_keys, args.keys_regex)
    if len(cand_keys) >= args.n_paraphrases:
        cand_keys = cand_keys[:args.n_paraphrases]
    include_original = (args.include_original.lower() == "true")
    chosen = [("instruction_original", item.instruction_original)] if include_original else []
    chosen += [(k, item.paraphrases[k]) for k in cand_keys]

    base, ft, tokenizer = build_model_and_tokenizer(
        args.base_model_name_or_path, args.ft_model_name_or_path, args.ft_lora_adapter, args.device
    )

    enc_list: List[Encoded] = []
    base_As: List[torch.Tensor] = []
    ft_As: List[torch.Tensor] = []
    base_runner = capture_attn(base, args.layer_index, args.device)
    ft_runner   = capture_attn(ft,   args.layer_index, args.device)

    for key, text in chosen:
        prompt = build_prompt_text(text, item.input_text)
        enc = encode_prompt(tokenizer, prompt, args.device, prompt_span="no_bos")
        A_base = base_runner(enc)
        A_ft   = ft_runner(enc)
        base_As.append(A_base)
        ft_As.append(A_ft)
        enc_list.append(enc)
        toks = humanize_tokens(enc.tokens, args.max_tokens)
        Path(outdir / f"tokens_prompt{item.prompt_count}_{key}.txt").write_text("\n".join(toks), encoding="utf-8")

    base_runner.hooks.remove_all(); ft_runner.hooks.remove_all()

    H = ft_As[0].shape[0]
    cos_per_head: List[float] = []
    for h in range(H):
        sims = []
        for enc, Ab, Af in zip(enc_list, base_As, ft_As):
            vb = flatten_crop_attn(Ab[h], enc.prompt_mask, args.max_tokens)
            vf = flatten_crop_attn(Af[h], enc.prompt_mask, args.max_tokens)
            cb = F.normalize(vb.float(), dim=0); cf = F.normalize(vf.float(), dim=0)
            sims.append(float(torch.clamp((cb * cf).sum(), -1.0, 1.0).item()))
        cos_per_head.append(float(np.mean(sims)))
    head_rank = sorted([(h, cos) for h, cos in enumerate(cos_per_head)], key=lambda x: x[1])
    df_rank = pd.DataFrame([{"head": h, "base_ft_cosine_mean": cos} for h,cos in head_rank])
    df_rank.to_csv(outdir / "head_rank_by_change.csv", index=False)

    topk = min(args.select_heads_topk, H)
    selected_heads = [h for h,_ in head_rank[:topk]]

    head_rows = []
    for h in selected_heads:
        n = len(enc_list)
        cols = 4 if n > 4 else n
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.0*cols, 3.0*rows))
        if not isinstance(axes, np.ndarray): axes = np.array([[axes]])
        axes = axes.reshape(rows, cols)

        diag_w1_vals = []; diag_w2_vals = []
        top_across_count: Dict[str, int] = {}

        for i, (key, _text) in enumerate(chosen):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            enc = enc_list[i]
            A = ft_As[i][h]
            idx = enc.prompt_mask.nonzero(as_tuple=True)[0]
            if args.max_tokens is not None and idx.numel() > args.max_tokens:
                idx = idx[:args.max_tokens]
            idx = idx.to(A.device)
            A_crop = A[idx][:, idx].float()

            im = ax.imshow(A_crop.numpy(), origin="lower", aspect="auto", interpolation="nearest")
            ax.set_title(key, fontsize=9)
            toks = humanize_tokens([enc.tokens[j.item()] for j in idx], None)
            step = max(1, len(toks)//12)
            xt = list(range(0, len(toks), step))
            yt = list(range(0, len(toks), step))
            ax.set_xticks(xt); ax.set_xticklabels([toks[j] for j in xt], rotation=90, fontsize=6)
            ax.set_yticks(yt); ax.set_yticklabels([toks[j] for j in yt], fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            d1, d2 = diagonality_ratios(A_crop)
            diag_w1_vals.append(d1); diag_w2_vals.append(d2)

            col_mass = A_crop.mean(dim=0)
            last_row = A_crop[-1] if A_crop.shape[0] > 0 else torch.zeros_like(col_mass)
            topK = min(10, A_crop.shape[0])
            top_idx = torch.topk(col_mass, k=topK).indices.cpu().tolist()
            top_tokens = [toks[j] for j in top_idx]
            for t in top_tokens:
                top_across_count[t] = top_across_count.get(t, 0) + 1

            df_pp = pd.DataFrame({
                "token": toks,
                "incoming_mass": col_mass.cpu().numpy(),
                "last_token_attn": last_row.cpu().numpy()
            }).sort_values("incoming_mass", ascending=False)
            df_pp.to_csv(outdir / f"head{h}_top_targets_{key}.csv", index=False, encoding="utf-8")

        for j in range(len(chosen), rows*cols):
            r, c = divmod(j, cols)
            axes[r, c].axis("off")

        fig.suptitle(f"Layer {args.layer_index} — FT attention — head {h}", y=0.995, fontsize=11)
        fig.tight_layout()
        fig.savefig(outdir / f"head{h}_attn_grid.png", dpi=180)
        plt.close(fig)

        top_across_df = pd.DataFrame(
            sorted(top_across_count.items(), key=lambda kv: kv[1], reverse=True),
            columns=["token_string","count_across_paraphrases"]
        )
        top_across_df.to_csv(outdir / f"head{h}_top_targets_across_paraphrases.csv", index=False, encoding="utf-8")

        head_rows.append({
            "head": h,
            "base_ft_cosine_mean": float(df_rank.loc[df_rank["head"]==h, "base_ft_cosine_mean"].values[0]),
            "diag_ratio_w1_mean": float(np.mean(diag_w1_vals)) if diag_w1_vals else np.nan,
            "diag_ratio_w2_mean": float(np.mean(diag_w2_vals)) if diag_w2_vals else np.nan,
            "examples_top_tokens": ", ".join(top_across_df.head(8)["token_string"].tolist())
        })

    pd.DataFrame(head_rows).to_csv(outdir / "head_stats.csv", index=False, encoding="utf-8")

    report = []
    report.append(f"# FT attention inspection (layer {args.layer_index})")
    report.append(f"- prompt_count: **{item.prompt_count}**")
    report.append(f"- paraphrases used ({len(chosen)}): " + ", ".join([k for k,_ in chosen]))
    report.append(f"- selected heads (most changed vs BASE): " + ", ".join([str(h) for h in selected_heads]))
    report.append("")
    report.append("## Files")
    report.append("- `head_rank_by_change.csv`")
    report.append("- `head_stats.csv`")
    report.append("- `head{H}_attn_grid.png` (one per selected head)")
    report.append("- `head{H}_top_targets_{KEY}.csv` (per paraphrase)")
    report.append("- `head{H}_top_targets_across_paraphrases.csv`")
    Path(outdir / "report.md").write_text("\n".join(report), encoding="utf-8")

if __name__ == "__main__":
    main()
