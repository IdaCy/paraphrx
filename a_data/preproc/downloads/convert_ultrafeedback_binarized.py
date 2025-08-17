"""
python3 a_data/preproc/downloads/convert_ultrafeedback_binarized.py \
  --split train_sft \
  --out_prompts_json a_data/ultrafeedback_binarized/ufb_train_prompts.json \
  --out_answers_json a_data/ultrafeedback_binarized/ufb_train_answers.json
"""

import argparse, json
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train_sft",
                    help="One of: train_sft, test_sft (from ultrafeedback_binarized)")
    ap.add_argument("--revision", default=None,
                    help="Optional HF revision hash if you want the older version")
    ap.add_argument("--out_prompts_json", required=True)
    ap.add_argument("--out_answers_json", required=True)
    args = ap.parse_args()

    # Load the dataset split
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized",
                      split=args.split,
                      revision=args.revision) if args.revision \
        else load_dataset("HuggingFaceH4/ultrafeedback_binarized",
                          split=args.split)

    # Build format:
    #    - instruction_original := the user turn from the CHOSEN conversation (role == 'user')
    #    - output               := the assistant turn from the CHOSEN conversation (role == 'assistant')
    #    - input                := "" (no separate input field in this dataset)
    prompts, answers = [], []
    pc = 1
    for ex in ds:
        chosen = ex.get("chosen", None)
        if not chosen or len(chosen) < 2:
            continue
        # chosen is a list of dicts: [{"role":"user","content":...}, {"role":"assistant","content":...}]
        try:
            user_msg = next(m["content"] for m in chosen if m.get("role") == "user")
            asst_msg = next(m["content"] for m in chosen if m.get("role") == "assistant")
        except StopIteration:
            # fallback: try 'messages' (same content duplicated)
            msgs = ex.get("messages", [])
            user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            asst_msg = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
            if not user_msg or not asst_msg:
                continue

        prompts.append({
            "prompt_count": pc,
            "instruction_original": user_msg.strip(),
            "input": "",
            "output": asst_msg.strip()
        })
        # answers JSON is only needed if run with use_prompt_output_as_answer=False
        answers.append({
            "prompt_count": pc,
            "instruction_original": asst_msg.strip()
        })
        pc += 1

    with open(args.out_prompts_json, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    with open(args.out_answers_json, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(prompts)} examples to {args.out_prompts_json} and {args.out_answers_json}")

if __name__ == "__main__":
    main()
