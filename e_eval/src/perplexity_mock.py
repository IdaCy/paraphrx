# -*- coding: utf-8 -*-
"""
    python e_eval/src/perplexity_mock.py \
        --input c_assess_inf/output/mmlu_answer_scores/gemma-2-2b-it.json \
        --output e_eval/data/ppl_mock_mmlu_gemma-2-2b-it.json

1. Loads a JSON file where each entry has 'prompt_count',
   multiple paraphrase-type score arrays, and 'instruction_original'
2. Extracts for each entry:
   - prompt_count
   - answer_scores (the 10-item list from 'instruction_original')
   - a mocked perplexity value (random float between 100 and 500)
3. Writes out a JSON array for the 'original-score files' input to paraphrase_stats.py
"""
import json
import random
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate original-prompt score JSON for paraphrase_stats.py"
    )
    parser.add_argument(
        "--input", required=True, help="Path to combined scores JSON"
    )
    parser.add_argument(
        "--output", required=True, help="Path to write original_scores.json"
    )
    args = parser.parse_args()

    # Load the full scores file
    with open(args.input, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    original_scores = []
    for entry in full_data:
        # Extract the 10-item list under 'instruction_original'
        scores = entry.get("instruction_original")
        if not isinstance(scores, list) or len(scores) != 10:
            raise ValueError(
                f"Entry for prompt_count={entry.get('prompt_count')} "
                "is missing a valid 'instruction_original' score list."
            )
        record = {
            "prompt_count": entry["prompt_count"],
            "answer_scores": scores,
            # Mock perplexity: random float between 100 and 500
            "perplexity": round(random.uniform(100.0, 500.0), 2)
        }
        original_scores.append(record)

    # Write out to the output JSON
    with open(args.output, 'w', encoding='utf-8') as out_f:
        json.dump(original_scores, out_f, indent=2)

    print(f"Wrote {len(original_scores)} records to {args.output}")


if __name__ == '__main__':
    main()
