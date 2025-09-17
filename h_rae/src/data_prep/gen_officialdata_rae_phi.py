#!/usr/bin/env python3
"""
Batched local inference for microsoft/Phi-3.5-mini-instruct (or a local snapshot),
writing outputs in the minimal structure the user requested.

OUTPUT FORMAT EXAMPLE (per item):
{
  "prompt_count": 1,
  "instruction_original": "<model answer>",
  "instruct_1": "<model answer>",
  "instruct_2": "<model answer>",
  ...
}

python3 h_rae/src/data_prep/gen_officialdata_rae_phi_batched.py \
    --prompts h_rae/data/rae_official/RobustAlpacaEval_converted.json \
    --output  h_rae/data/baseline/phi_answers_rae.json \
    --model   f_finetune/model/phi35 \
    --log-name AnswerGenPhi35 \
    --max-samples 0 \
    --batch-size 64 \
    --max-new-tokens 256 --temperature 0.0
"""

import json, argparse, time, random, re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Chat template (Phi-3.5-mini-instruct)
def build_phi35_chat(
    user_prompt: str,
    system_prompt: str = (
        "You are a concise, reliable assistant. "
        "If the prompt is multiple choice, answer with the single best option letter "
        "plus a brief explanation. Keep answers tight and directly responsive."
    ),
) -> str:
    return (
        "<|system|>\n" + system_prompt + "<|end|>\n"
        "<|user|>\n" + user_prompt.strip() + "<|end|>\n"
        "<|assistant|>\n"
    )

# Discover ALL prompts in an item (not limited by prompt_count)
_instruct_pat  = re.compile(r"^instruct_(\d+)$")
_intruct_pat   = re.compile(r"^intruct_(\d+)$")  # typo fallback

def list_all_prompt_fields(item: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Returns a list of (canonical_key, prompt_text) to generate for:
      - 'instruction_original' if present and non-empty
      - every 'instruct_N' present
      - every 'intruct_N' (typo) present, mapped to canonical 'instruct_N' if that canonical key is absent
    Sorted by field order: instruction_original first, then instruct_1..instruct_10.. etc.
    """
    fields: Dict[str, str] = {}

    # instruction_original
    if item.get("instruction_original"):
        fields["instruction_original"] = item["instruction_original"]

    # collect instruct_N and intruct_N
    tmp: Dict[int, str] = {}
    for k, v in item.items():
        if not v:
            continue
        m = _instruct_pat.match(k)
        if m:
            idx = int(m.group(1))
            tmp[idx] = v

    # fill from intruct_N if instruct_N missing
    for k, v in item.items():
        if not v:
            continue
        m = _intruct_pat.match(k)
        if m:
            idx = int(m.group(1))
            if idx not in tmp:
                tmp[idx] = v

    # order by index
    for idx in sorted(tmp.keys()):
        fields[f"instruct_{idx}"] = tmp[idx]

    # return as ordered list of tuples
    return [(k, fields[k]) for k in (["instruction_original"] if "instruction_original" in fields else [])] + \
           [(k, fields[k]) for k in fields.keys() if k != "instruction_original"]

# Special token ids
def get_special_ids(tokenizer):
    end_tok = "<|end|>"
    end_id = tokenizer.convert_tokens_to_ids(end_tok)
    if end_id is None or end_id == tokenizer.unk_token_id:
        end_id = tokenizer.eos_token_id
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else end_id
    return {"eos": end_id, "pad": pad_id}

# Speed knobs for A100
def enable_a100_fastpath():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        from torch.backends.cuda import sdp_kernel
        # Prefer flash + mem-efficient; disable math fallback
        sdp_kernel.enable_flash_sdp(True)
        sdp_kernel.enable_mem_efficient_sdp(True)
        sdp_kernel.enable_math_sdp(False)
    except Exception:
        pass
    return None  # set to a compile wrapper if you choose to use torch.compile

# MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True, help="Local model dir (e.g., f_finetune/model/phi35) or HF repo id")
    ap.add_argument("--log-name", default="AnswerGenPhi35")
    ap.add_argument("--max-samples", type=int, default=0, help="Limit items; 0 = all")
    ap.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=200, help="Checkpoint every N generations")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[{args.log_name}] Loading model/tokenizer from: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)

    enable_a100_fastpath()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    try:
        model.config.attn_implementation = "sdpa"
    except Exception:
        pass

    ids = get_special_ids(tokenizer)

    data = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")
    if args.max_samples and args.max_samples > 0:
        data = data[:args.max_samples]

    # Build tasks: (item_idx, target_key, chat_prompt) over ALL prompt fields
    tasks: List[Tuple[int, str, str]] = []
    for item_idx, item in enumerate(data):
        for target_key, prompt_text in list_all_prompt_fields(item):
            user_prompt = prompt_text
            # If the item includes an "input" field, append it (kept from your previous logic)
            if item.get("input"):
                user_prompt = f"{user_prompt}\n\nAdditional input:\n{item['input']}"
            chat = build_phi35_chat(user_prompt)
            tasks.append((item_idx, target_key, chat))

    # Length bucketing to reduce padding
    lengths = []
    for i, (_, _, chat) in enumerate(tasks):
        l = len(tokenizer(chat, add_special_tokens=False)["input_ids"])
        lengths.append((l, i))
    lengths.sort(reverse=True)  # longest first
    ordered_tasks = [tasks[i] for _, i in lengths]

    # Prepare output skeleton (minimal structure)
    out_items: List[Dict[str, Any]] = []
    for item in data:
        out_obj: Dict[str, Any] = {"prompt_count": item.get("prompt_count", 0)}
        # Precreate keys with None to keep ordering stable (optional)
        if item.get("instruction_original"):
            out_obj["instruction_original"] = None
        # Add discovered instruct_N slots
        discovered = set()
        for k in item.keys():
            m1 = _instruct_pat.match(k)
            m2 = _intruct_pat.match(k)
            if m1:
                discovered.add(int(m1.group(1)))
            elif m2:
                discovered.add(int(m2.group(1)))
        for idx in sorted(discovered):
            out_obj[f"instruct_{idx}"] = None
        out_items.append(out_obj)

    total = len(ordered_tasks)
    done = 0
    t0_all = time.time()

    # Batched generation loop
    while done < total:
        batch = ordered_tasks[done : min(done + args.batch_size, total)]
        item_indices = [t[0] for t in batch]
        target_keys  = [t[1] for t in batch]
        chats        = [t[2] for t in batch]

        t0 = time.time()
        with torch.no_grad():
            enc = tokenizer(
                chats,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            ).to(model.device)

            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=ids["eos"],
                pad_token_id=ids["pad"],
            )

        # Slice off prompts row-wise and decode
        input_len = enc["input_ids"].shape[1]
        gens = out[:, input_len:]
        texts = tokenizer.batch_decode(gens, skip_special_tokens=True)
        dt_ms = int((time.time() - t0) * 1000)

        # Write into minimal per-item objects under the target keys
        for i in range(len(batch)):
            out_items[item_indices[i]][target_keys[i]] = texts[i].strip()

        done += len(batch)
        if done % max(16, args.batch_size) == 0 or done == total:
            el = int(time.time() - t0_all)
            print(f"[{args.log_name}] {done}/{total} prompts | last_batch {dt_ms} ms | elapsed {el}s", flush=True)

        # periodic checkpoint (optional but helpful on long runs)
        if args.save_every and (done % args.save_every == 0 or done == total):
            Path(args.output).write_text(json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8")

    # Final write
    Path(args.output).write_text(json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.log_name}] DONE {done} prompts in {int(time.time()-t0_all)}s -> {args.output}", flush=True)

if __name__ == "__main__":
    main()
