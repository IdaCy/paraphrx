#!/usr/bin/env python
"""
python check_inf.py \
      --model_dir  f_finetune/model \
      --lora_dir   f_finetune/outputs_bkt1_2/final \
      --data_json  f_finetune/data/alpaca_gemma-2-2b-it.json \
      --bucket_spec 1,2 \
      --val_idx f_finetune/outputs_bkt1_2/val_idx.json \
      --num 10 \
      --compare_base
"""
import argparse, json, random, time, math, sys, torch, datetime as dt
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# CLI
ap = argparse.ArgumentParser()
ap.add_argument("--model_dir",  required=True)
ap.add_argument("--lora_dir",   required=True)
ap.add_argument("--data_json",  required=True)
ap.add_argument("--bucket_spec", required=True,
                help='Same syntax as training, e.g. "1-3,5"')
ap.add_argument("--val_idx",    help="JSON list with held-out indices (optional)")
ap.add_argument("--num", type=int, default=5)
ap.add_argument("--compare_base", action="store_true",
                help="Also generate with the base model for reference")
args = ap.parse_args()

# helper
def parse_spec(spec):
    res = set()
    for part in spec.split(","):
        if "-" in part:
            a,b = map(int, part.split("-")); res.update(range(a, b+1))
        elif part.strip():
            res.add(int(part))
    return sorted(res)

def alpaca_prompt(item):
    head = f"### Instruction:\n{item['instruction_original']}\n\n"
    if item.get("input"): head += f"### Input:\n{item['input']}\n\n"
    return head + "### Response:\n"

def load_idx(path):
    try: return json.loads(Path(path).read_text())
    except FileNotFoundError: return None

# load data
wanted_bk   = set(parse_spec(args.bucket_spec))
with open(args.data_json) as fp:
    raw = json.load(fp)

# pick rows whose original bucket is in wanted_bk
rows = [r for r in raw if any(int(p.get("bucket",0)) in wanted_bk
                              for p in r.get("paraphrases", []))]

# honour stored validation indices if supplied
if args.val_idx:
    idx_keep = set(load_idx(args.val_idx))
    rows = [r for i,r in enumerate(rows) if i in idx_keep]

assert rows, "No rows left after filtering; check bucket_spec/val_idx"

sample = random.sample(rows, k=min(args.num, len(rows)))
print(f"[{dt.datetime.now().strftime('%F %T')}] "
      f"Evaluating {len(sample)} examples | buckets {wanted_bk}")

# load models
tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
base = AutoModelForCausalLM.from_pretrained(args.model_dir,
                                            quantization_config=bnb,
                                            device_map="auto")
ft   = PeftModel.from_pretrained(base, args.lora_dir)
ft.eval()

# generation loop
for ex in sample:
    prompt = alpaca_prompt(ex)
    inp     = tok(prompt, return_tensors="pt").to(ft.device)

    t0 = time.time()
    out_ft = ft.generate(**inp, max_new_tokens=160, do_sample=False)
    resp_ft = tok.decode(out_ft[0], skip_special_tokens=True
              ).split("### Response:")[-1].strip()
    dt_ft = time.time() - t0

    if args.compare_base:
        t0 = time.time()
        out_b = base.generate(**inp, max_new_tokens=160, do_sample=False)
        resp_b = tok.decode(out_b[0], skip_special_tokens=True
                  ).split("### Response:")[-1].strip()
        dt_b = time.time() - t0

    print("="*88)
    print("INSTRUCTION:", ex["instruction"][:140])
    if ex["input"]: print("INPUT      :", ex["input"][:140])
    print("REFERENCE   :", ex["output"][:200].replace("\n"," ") )
    print(f"FINE-TUNED  ({dt_ft:.1f}s):", resp_ft[:200].replace("\n"," "))
    if args.compare_base:
        print(f"BASE        ({dt_b :.1f}s):", resp_b[:200].replace("\n"," "))

print("="*88)
print("done.")

