# Paraphrase Investigations

The Limit of LLMs with paraphrased style difference

## Sub Project 1: How do you want to talk to an LLM to have the best outcome - PRODUCT

- big comparison dashboard
- flagging style
- people insert something -> ranked how good result -> translate from their style to best style!!

## Data

Initial data from: https://huggingface.co/datasets/tatsu-lab/alpaca  

## Versions 

Too much content - moved over to categories.md!  

Generating different versions with gemini-2.5  

Test file: politeness, with 5 levels

## Inference Checks

Generating the output with for every type of instructions:
```
python c_assess_inf/src/run_inference.py \
       b_tests/data/alpaca_10_politeness.json \
       c_assess_inf/output/results_test.json \
       --model google/gemma-2b-it \
       --temperature 0 \
       --max_tokens 256
```

## Evaluating each paraphrase answer with Gemini-2.5 using schema:

  ```
  {
    "is_correct": boolean,        // Did the answer fully satisfy the instruction?
    "score_0_to_5": integer,      // 0 = useless, 5 = perfect
    "explanation": string         // 1-2 sentence rationale
  }
  ```

-> Produces one JSON object per record whose keys follow the pattern
  `<PARAPHRASE_KEY>_eval`.
