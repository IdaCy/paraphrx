## README — `f_finetune/`

*LoRA fine-tuning pipeline for paraphrase-robustness experiments on Gemma-2-2B-IT*

---

###  Overview

run **five independent LoRA fine-tunes** (buckets 1, 1-2, 1-3, 1-4, 1-5) on paraphrase-annotated Alpaca/GSM8K/MMLU JSON files

teach a base instruction-tuned model (`gemma-2-2b-it`) to **stay consistent across diverse paraphrase styles**

([github.com/tloen/alpaca-lora][1])

---

###  Directory

```
f_finetune/
 ├─ data/
 │   └─ alpaca_gemma-2-2b-it.json
 ├─ src/
 │   ├─ data_prep.rs
 │   └─ finetuning.py
 ├─ model/       # local copy of google/gemma-2-2b-it
 ├─ outputs/     # LoRA adapter checkpoints per run
 ├─ Cargo.toml
 └─ README.md    # <— you are here
```

---

###  Dataset specication

Each JSON **list element** represents one original instruction plus *all* its paraphrases:

| Key                    | Type       | Notes                                                                                                                                                                                                      |
| ---------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt_count`         | int        | Primary ID                                                                                                                                                                                                |
| `instruction_original` | str        | Canonical wording                                                                                                                                                                                         |
| `input`                | str        | Optional extra context (may be empty)                                                                                                                                                                     |
| `output`               | str        | Reference answer (ground truth)                                                                                                                                                                           |
| `count_in_buckets`     | list\[int] | `[c₁,c₂,c₃,c₄,c₅]` for bucket sizes                                                                                                                                                                       |
| `paraphrases`          | list\[obj] | Includes **original** + all `instruct_*`. Each object holds:<br>• `instruct_type` <br>• `paraphrase`<br>• `answer` (Gemma’s reply)<br>• `task_score` (0-10)<br>• `ranking_for_buckets`<br>• `bucket` (1-5) |

-> follows quantile rule

---

### LoRA + 4-bit

the model is loaded once in **NF4 4-bit** using `bitsandbytes`, then frozen; a few million LoRA parameters are trained, letting a 24 GB GPU handle 2B weights

Hugging Face’s `peft` API plus `transformers.Trainer` is the normal recipe adopted by Google & Databricks Gemma tutorials

([huggingface.co][2])
([databricks.com][3], [huggingface.co][4])

---

###  Hyperparameter defaults

| Group           | Value                                                                          | Reasoning / sources                                                                                                                                     |
| --------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LoRA**        | `r=16`, `alpha=32`, `dropout=0.05`, target modules=`q,k,v,o,gate,up,down_proj` | “sweet spot” for ≤3 B models; `alpha ≃ 2 × r` heuristic; dropout recommended for style-mixed data ([magazine.sebastianraschka.com][5], [github.com][6]) |
| **Optimiser**   | AdamW β₁=0.9 β₂=0.95 ε=1e-6, weight-decay 0.0                                  | Like Databricks & Google examples ([databricks.com][3], [huggingface.co][4])                                                                         |
| **LR schedule** | 2e-4, cosine, 5 % warm-up                                                      | Outperforms linear on short LoRA runs ([databricks.com][3])                                                                                             |
| **Epochs**      | 3                                                                              | Diminishing returns beyond 3-4 on comparable datasets ([magazine.sebastianraschka.com][5])                                                               |
| **Batching**    | per-GPU 4, grad-accum 4 → eff 16                                               | Fits 24 GB with 4-bit weights; community reference configs ([huggingface.co][4])                                                                        |
| **Precision**   | `--bf16` optional                                                              | Faster on Ampere/Hopper while matching fp16 quality ([huggingface.co][4])                                                                               |

All knobs are CLI-overrideable.

---

### Use

```bash
cd f_finetune
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Core packages: transformers[torch]>=4.41, peft>=0.11,
#                bitsandbytes>=0.44, datasets,
#                loguru, wandb, accelerate
```

ran it with 1x A100, VRAM 40GB, on ESE_BORG

CLI use:

```bash
python finetune_paraphrx.py \
  --run_name bucket1 \
  --data_paths data/alpaca_gemma-2-2b-it.json \
  --buckets 1 \
  --output_dir outputs \
  --wandb_project paraphrx \
  --bf16        # optional
```

Main flags:

| Flag                          | Default        | Description                                                          |
| ----------------------------- | -------------- | -------------------------------------------------------------------- |
| `--data_paths`                | *required*     | One or more JSON files (space-separated)                            |
| `--buckets`                   | `1`            | Comma-separated list or range (`1-3`)                               |
| `--run_name`                  | auto timestamp | Prefix for log file and output dir                                  |
| `--output_dir`                | `outputs/`     | Where adapters & Trainer state go                                   |
| `--use_paraphrase_answer`     | *false*        | Train on paraphrase-specific `answer` instead of canonical `output` |
| `--bf16 / --fp16`             | off / on       | Mixed precision toggle                                              |
| `--r --alpha --lora_dropout`  | see §5         | Hyperparameters                                                     |
| `--lr --num_train_epochs ...` |               | Standard Trainer args                                               |

Run the five experiments e.g.:

```bash
for b in 1 1-2 1-3 1-4 1-5; do
  python finetune_paraphrx.py --buckets $b --run_name run_$b \
         --data_paths data/alpaca_gemma-2-2b-it.json
done
```

---

###  Plan (demoted to description)

1. **Logger boot-strap**  
   *Loguru* creates `logs/<run>_<yyyy-mm-dd_HH-MM-SS>.log`; every event has a UTC timestamp+level

2. **Dataset load & bucket filter**

   * Reads all JSONs into a single `datasets.Dataset`
   * Filters `paraphrases` by `bucket` ∈ chosen list while **always keeping** their shared `instruction_original`, `input`, `output`
   * 80/20 random split (seeded by `--seed`)

3. **Prompt template**
   Builds a single string:

   ```
   ### Instruction:
   {instruction}
   ### Input:
   {input}
   ### Response:
   ```

   followed by `<eos>` token.
   This follows the Alpaca-LoRA format -([github.com/tloen/alpaca-lora][1])

4. **Tokenizer & 4-bit model load**

   ```python
   model = AutoModelForCausalLM.from_pretrained(
       local_dir,
       quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                              bnb_4bit_compute_dtype=torch.bfloat16,
                                              bnb_4bit_use_double_quant=True,
                                              bnb_4bit_quant_type="nf4"))
   model.gradient_checkpointing_enable()
   ```

5. **PEFT-LoRA wrapping**

   ```python
   peft_config = LoraConfig(r=args.r, lora_alpha=args.alpha,
                            target_modules=args.target_modules.split(","),
                            lora_dropout=args.lora_dropout,
                            bias="none", task_type="CAUSAL_LM")
   model = get_peft_model(model, peft_config)
   model.print_trainable_parameters()   # logged
   ```

6. **Trainer setup**

   * Data-collator pads to max in batch
   * `transformers.Trainer` with cosine schedule, eval 20 steps
   * WandB callback (optional)

7. **Training loop**
   *Epoch-loss, eval-loss, ppl, LR, memory* logged & pushed to WandB

   * Early stopping optional (`--patience`)

8. **Saving**  
   Saves **LoRA adapters only** (`adapter_config.json`, `adapter_model.bin`) plus a `trainer_state.json` under `outputs/<run_name>/`  
   \+ symlinks latest log

---

### + to GSM8K / MMLU

additional paraphrase JSONs in `data/` -> listing them:

```bash
python finetune_paraphrx.py --data_paths \
  data/alpaca_gemma-2-2b-it.json data/gsm8k_paraphrases.json
```

-> mergin rows + keeping bucket logic per-prompt

---

###  Evaluation (still plan)

After training:

```bash
python evaluate_paraphrx.py \
  --adapter outputs/run_1/checkpoint-??? \
  --eval_json data/alpaca_gemma-2-2b-it.json \
  --buckets all
```

`evaluate_paraphrx.py` will:

* run the base Gemma with LoRA weights merged (`model.merge_and_unload()`),
* compute exact-match / BLEU on `output`,
* compute robustness gap between paraphrase buckets.

###   Extra/Refs

* The script is Apache-2.0; underlying Gemma weights are under the Google AI license.
* Ideas & hyper-params draw heavily on open-source LoRA work by the Alpaca-LoRA team ([github.com/tloen/alpaca-lora][1]), Hugging Face QLoRA researchers ([huggingface.co][2]), Databricks guides ([databricks.com][3]), Sebastian Raschka’s best-practice notes ([magazine.sebastianraschka.com][5]), and vLLM LoRA discussions ([github.com][6]).


[1]: https://github.com/tloen/alpaca-lora "tloen/alpaca-lora: Instruct-tune LLaMA on consumer hardware"
[2]: https://huggingface.co/blog/4bit-transformers-bitsandbytes "Making LLMs even more accessible with bitsandbytes, 4-bit ..."
[3]: https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms "Efficient Fine-Tuning with LoRA for LLMs | Databricks Blog"
[4]: https://huggingface.co/PranavKeshav/event-planner-gemma-4bit "PranavKeshav/event-planner-gemma-4bit · Hugging Face"
[5]: https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms "Practical Tips for Finetuning LLMs Using LoRA (Low-Rank Adaptation)"
[6]: https://github.com/vllm-project/vllm/issues/2816 "VLLM Multi-Lora with embed_tokens and lm_head in adapter weights"
