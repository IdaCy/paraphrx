# Paraphrase Investigations

The Limit of LLMs with paraphrased style difference

## Sub Project 1: How do you want to talk to an LLM to have the best outcome

- Development big comparison dashboard
- Flagging style
- people insert something -> ranked how good result -> translate from their style to best style

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

## Metrics
1. Task Fulfilment / Relevance - Does it respond to every part of the prompt? Did it wander off-topic or over-answer?
2. Usefulness & Actionability - Does it translate abstract ideas into concrete advice, examples, or next steps?
3. Factual Accuracy & Verifiabiliy - Are the statements factually correct (no hallucinations)? If it cites sources or internal steps, do those match the final claims?
4. Efficiency / Depth & Completeness - Does it avoid unnecessary verbosity or excessive brevity? Does it cover the key angles, edge-cases, and typical follow-ups? Could the user act on it without having to ask “what about X?”
5. Reasoning Quality / Transparency - Are the steps implicitly or explicitly sound? If uncertain, does it flag that uncertainty instead of bluffing?
6. Tone & Likeability - Is the style friendly and respectful, matching the user’s vibe? Would you enjoy a longer conversation in this voice?
7. Adaptation to Context - Does it use any relevant info the user has shared (location, preferences, prior messages) appropriately?
8. Safety & Bias Avoidance - Does it steer clear of harmful or disallowed content? Does it acknowledge and mitigate possible bias?
9. Structure & Formatting & UX Extras - Is the writing logically ordered and easy to skim? Are lists, code blocks, tables, or rich widgets used when, but only when they genuinely improve readability or utility?
10. Creativity - Does the answer make clever, non-obvious connections that you wouldn’t get from a quick Google search? Or, does it remix ideas, metaphors, or examples in a fresh way rather than serving boilerplate?
