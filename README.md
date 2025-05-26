# Phraraphrase Investigations

## Data

Initial data from: https://huggingface.co/datasets/tatsu-lab/alpaca  

### Versions 

too much - moved over to categories.md

### Evaluating each paraphrase answer with Gemini-2.5 using the *grading schema*:

  ```
  {
    "is_correct": boolean,        // Did the answer fully satisfy the instruction?
    "score_0_to_5": integer,      // 0 = useless, 5 = perfect
    "explanation": string         // 1-2 sentence rationale
  }
  ```

* **Produces one JSON object per record** whose keys follow the pattern
  `<PARAPHRASE_KEY>_eval`.
