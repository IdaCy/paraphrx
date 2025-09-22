#!/usr/bin/env python3
"""
Run RobustAlpacaEval on Gemma-2 (or compatible) with optional LoRA,
optional weight merge+save, and judge-friendly decoding modes

Outputs a JSON list: [{"instruction","output","group_id","variant","model"}]
"""

import argparse, json, sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


META_DEFAULT = (
    "Answer concisely and directly. Focus on task semantics; ignore stylistic tone cues. "
    "Do not add preambles or apologies. End after the answer."
)


def load_robust_jsonl(path: str) -> List[Dict[str, Any]]:
    flat = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            gid = int(item["index"])
            flat.append({"group_id": gid, "variant": "original", "prompt": item["instruction"]})
            for j, p in enumerate(item.get("paraphrases", [])):
                flat.append({"group_id": gid, "variant": f"paraphrase_{j}", "prompt": p})
    return flat


def build_chat(tokenizer, user_text: str, meta: str, use_system: bool) -> str:
    if use_system:
        messages = [
            {"role": "system", "content": meta},
            {"role": "user", "content": user_text},
        ]
    else:
        # Put meta into user turn (safe for templates that ignore 'system')
        messages = [{"role": "user", "content": f"{meta}\n\n{user_text}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


import re

ROLE_LINE = re.compile(r'^\s*(?:assistant|model|user|system)\s*[:\n]\s*', re.IGNORECASE)

def clean_output(text: str, meta: Optional[str] = None, prompt: Optional[str] = None) -> str:
    # def clean_output(text: str, meta: str | None = None) -> str:
    t = text.lstrip()

    # Drop leading role headers (possibly repeated)
    while True:
        m = ROLE_LINE.match(t)
        if not m:
            break
        t = t[m.end():]

    # If the meta got echoed at the top, remove it (with or without a blank line after)
    if meta:
        low = t.lower()
        mlow = meta.lower()
        if low.startswith(mlow):
            # Prefer cutting at first blank line if present
            parts = t.split("\n\n", 1)
            if len(parts) == 2 and parts[0].strip().lower().startswith(mlow):
                t = parts[1].lstrip()
            else:
                t = t[len(meta):].lstrip()

    # If the *user prompt* was echoed, remove it (also handle simple quoted variants)
    if prompt:
        p = prompt.strip()
        for cand in (p, f'"{p}"', f"'{p}'"):
            if t.startswith(cand):
                t = t[len(cand):].lstrip()
                break
        else:
            # Heuristic: if the first block has high token overlap with the prompt, drop up to the first blank line
            first_block = t.split("\n\n", 1)[0]
            if first_block and len(first_block) > 40:
                pw = set(p.lower().split())
                fw = set(first_block.lower().split())
                if pw and (len(pw & fw) / max(1, len(pw))) > 0.6:
                    t = t[len(first_block):].lstrip()

    # Drop role headers again in case echo/strip revealed them
    while True:
        m = ROLE_LINE.match(t)
        if not m:
            break
        t = t[m.end():]

    # Trim polite filler
    for lead in ("Sure, ", "Sure.", "Of course, ", "Of course.", "Certainly, ", "Certainly."):
        if t.startswith(lead):
            t = t[len(lead):].lstrip()

    return t.strip()


def load_model_and_tok(base: str, lora: str, merge: bool, dtype: str, device_map: str):
    torch_dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        base, device_map=device_map, trust_remote_code=True, torch_dtype=torch_dtype
    )
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # after building tokenizer
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    sot = tok.convert_tokens_to_ids("<start_of_turn>")
    eos_ids = [tok.eos_token_id]
    for tid in (eot, sot):
        if isinstance(tid, int) and tid >= 0:
            eos_ids.append(tid)

    if lora:
        try:
            from peft import PeftModel
        except Exception as e:
            print("peft is required when --lora_path is set:", e, file=sys.stderr)
            sys.exit(1)
        model = PeftModel.from_pretrained(model, lora, is_trainable=False)
        if merge:
            # Merge in higher precision to avoid quality loss
            if torch_dtype == "auto":
                # keep current; otherwise force bf16 which is typical on GPU
                pass
            print("Merging LoRA weights into base…", file=sys.stderr)
            model = model.merge_and_unload()
            print("Merge complete.", file=sys.stderr)
    model.eval()
    return model, tok


def gen_once(
    model, tok, rendered_batch: List[str],
    decode: str, max_new: int, rep_pen: float,
    temperature: float, top_p: float,
    penalty_alpha: float, csearch_topk: int,
    eos_ids: List[int], batch_size: int,
    src_prompts: Optional[List[str]] = None
) -> List[str]:
    outs: List[str] = []
    for i in tqdm(range(0, len(rendered_batch), batch_size), desc="Generating"):
        chunk = rendered_batch[i:i+batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True).to(model.device)
        input_lens = enc["attention_mask"].sum(dim=1)
        gen_kwargs = dict(
            max_new_tokens=max_new,
            eos_token_id=eos_ids,
            pad_token_id=tok.eos_token_id,
            repetition_penalty=rep_pen,     # e.g. 1.08–1.12
            no_repeat_ngram_size=3,
            use_cache=True,
        )

        if decode == "greedy":
            gen_kwargs.update(dict(do_sample=False))
        elif decode == "contrastive":
            # Contrastive Search (Li et al.), often strong for small models
            gen_kwargs.update(dict(do_sample=False, penalty_alpha=penalty_alpha, top_k=csearch_topk))
        elif decode == "beam":
            gen_kwargs.update(dict(do_sample=False, num_beams=4, num_return_sequences=1, length_penalty=0.3))
        else:  # "sample" (use sparingly for robustness)
            gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))

        with torch.inference_mode():
            out = model.generate(**enc, **gen_kwargs)
        for b in range(out.shape[0]):
            gen_ids = out[b, input_lens[b]:]
            raw = tok.decode(gen_ids, skip_special_tokens=True)
            prompt_text = None
            if src_prompts is not None:
                prompt_text = src_prompts[i + b]
            outs.append(clean_output(raw, meta=META_DEFAULT, prompt=prompt_text))
        del enc, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    return outs


def maybe_save_merged(model, save_dir: str, tok):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    print(f"Merged model saved to: {save_dir}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_jsonl", required=True)
    ap.add_argument("--base_model", default="google/gemma-2-2b-it")
    ap.add_argument("--lora_path", default="")
    ap.add_argument("--merge_lora", action="store_true")
    ap.add_argument("--save_merged_to", default="", help="Optional dir to save merged weights")
    ap.add_argument("--dtype", choices=["auto","bf16","fp16","fp32"], default="auto")
    ap.add_argument("--device_map", default="auto")

    ap.add_argument("--decode", choices=["greedy","contrastive","beam","sample"], default="contrastive")
    ap.add_argument("--penalty_alpha", type=float, default=0.6)  # contrastive only
    ap.add_argument("--csearch_topk", type=int, default=6)       # contrastive only
    ap.add_argument("--max_new_tokens", type=int, default=224)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--use_system_meta", action="store_true")
    ap.add_argument("--meta_prompt", default=META_DEFAULT)

    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    # Load dataset
    flat = load_robust_jsonl(args.dataset_jsonl)
    print(f"Loaded {len(flat)} prompts (11× per case).", file=sys.stderr)

    # Model
    model, tok = load_model_and_tok(args.base_model, args.lora_path, args.merge_lora, args.dtype, args.device_map)

    # Optionally save merged
    if args.merge_lora and args.save_merged_to:
        maybe_save_merged(model, args.save_merged_to, tok)

    # Prepare prompts
    rendered = [build_chat(tok, ex["prompt"], args.meta_prompt, args.use_system_meta) for ex in flat]

    # EOS handling: include end_of_turn if present
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = list({tok.eos_token_id} | ({eot} if isinstance(eot, int) and eot > 0 else set()))

    # Generate
    outputs = gen_once(
        model, tok, rendered,
        decode=args.decode,
        max_new=args.max_new_tokens,
        rep_pen=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        penalty_alpha=args.penalty_alpha,
        csearch_topk=args.csearch_topk,
        eos_ids=eos_ids,
        batch_size=args.batch_size,
        src_prompts=[ex["prompt"] for ex in flat]
    )

    # Pack results
    result = []
    assert len(outputs) == len(flat)
    for ex, out in zip(flat, outputs):
        result.append({
            "instruction": ex["prompt"],
            "output": out,
            "group_id": ex["group_id"],
            "variant": ex["variant"],
            "model": (args.save_merged_to or args.lora_path or args.base_model),
        })

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(Path(args.out_json).with_suffix(".jsonl"), "w", encoding="utf-8") as f:
        for r in result:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(result)} generations to {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
