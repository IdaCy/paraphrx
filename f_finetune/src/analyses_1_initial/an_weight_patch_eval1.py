#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
weight_patch_eval.py — Actual run with weight patching (answer-level robustness)
"""

import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------
# Helper functions
# ------------------
def set_seed(seed=42):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def cosine_distance(a, b, eps=1e-8):
    a = F.normalize(a, dim=-1, eps=eps); b = F.normalize(b, dim=-1, eps=eps)
    return float(1 - (a*b).sum().item())

def text_embedding(model, tokenizer, text, device):
    enc = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, return_dict=True)
        last = out.hidden_states[-1][0]  # [T,D]
    return last.mean(dim=0)

def generate_answer(model, tokenizer, prompt, device, max_new_tokens=128):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    gen_ids = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# ------------------
# Weight patching
# ------------------
class BlockAccessor:
    def __init__(self, model, layer_index):
        self.block = model.model.layers[layer_index]
        self.attn = self.block.self_attn
        self.mlp = self.block.mlp
    def swap_attention_from(self, other):
        for n in ["q_proj","k_proj","v_proj","o_proj"]:
            getattr(self.attn,n).weight.data.copy_(getattr(other.attn,n).weight.data)
    def swap_mlp_from(self, other):
        for n in ["up_proj","gate_proj","down_proj"]:
            getattr(self.mlp,n).weight.data.copy_(getattr(other.mlp,n).weight.data)

def build_hybrids(base, ft, layer_index, device):
    import copy
    ha, hm = copy.deepcopy(base).to(device), copy.deepcopy(base).to(device)
    ba, fa = BlockAccessor(ha,layer_index), BlockAccessor(hm,layer_index)
    ft_blk = BlockAccessor(ft,layer_index)
    ba.swap_attention_from(ft_blk)
    fa.swap_mlp_from(ft_blk)
    ha.eval(); hm.eval()
    return ha, hm

# ------------------
# Main eval
# ------------------
def run(args):
    device = args.device
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16).to(device).eval()
    ft   = AutoModelForCausalLM.from_pretrained(args.ft_model, torch_dtype=torch.float16).to(device).eval()
    tok  = AutoTokenizer.from_pretrained(args.base_model)
    tok.pad_token = tok.eos_token

    ha, hm = build_hybrids(base, ft, args.layer_index, device)
    models = {"BASE":base,"FT":ft,"HYB_ATTN":ha,"HYB_MLP":hm}

    data = json.loads(Path(args.instructions_json).read_text())
    out_rows, metrics = [], []

    for obj in data[:args.max_items]:
        pid = obj["prompt_count"]
        instr = obj["instruction_original"]
        paraphrases = [v for k,v in obj.items() if k.startswith("instruct_")]

        for mname, model in models.items():
            # generate original
            ans_orig = generate_answer(model,tok,instr,device,args.max_new_tokens)
            emb_orig = text_embedding(model,tok,ans_orig,device)
            dlist=[]
            for para in paraphrases:
                ans_p = generate_answer(model,tok,para,device,args.max_new_tokens)
                emb_p = text_embedding(model,tok,ans_p,device)
                dlist.append(cosine_distance(emb_orig,emb_p))
            D_model = np.mean(dlist) if dlist else np.nan
            metrics.append({"prompt_count":pid,"model":mname,"D_model":D_model})

    df = pd.DataFrame(metrics)
    pivot = df.pivot(index="prompt_count",columns="model",values="D_model").reset_index()
    for m in ["FT","HYB_ATTN","HYB_MLP"]:
        pivot[f"DELTA_{m}"] = pivot["BASE"] - pivot[m]

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True,parents=True)
    pivot.to_csv(outdir/"answer_diff_of_diffs.csv",index=False)

    # summary
    means = {m: pivot[f"DELTA_{m}"].mean() for m in ["FT","HYB_ATTN","HYB_MLP"]}
    print("Mean Δ (D_base − D_model):", means)

    # plot
    plt.bar(means.keys(),means.values()); plt.ylabel("Δ (higher=better)")
    plt.title("Paraphrase robustness improvement vs BASE")
    plt.tight_layout(); plt.savefig(outdir/"summary.png")

    # write report
    with open(outdir/"report.md","w") as f:
        f.write(f"# Weight patching analysis (layer {args.layer_index})\n\n")
        f.write("## Mean Δ (D_base − D_model)\n")
        for k,v in means.items(): f.write(f"- {k}: {v:.4f}\n")
        f.write("\nSee `answer_diff_of_diffs.csv` and `summary.png` for details.\n")

# ------------------
# CLI
# ------------------
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--instructions_json",required=True)
    ap.add_argument("--base_model",required=True)
    ap.add_argument("--ft_model",required=True)
    ap.add_argument("--layer_index",type=int,default=6)
    ap.add_argument("--max_items",type=int,default=30)
    ap.add_argument("--max_new_tokens",type=int,default=128)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir",required=True)
    run(ap.parse_args())
