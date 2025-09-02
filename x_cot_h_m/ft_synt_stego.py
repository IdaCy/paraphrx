#!/usr/bin/env python3
"""
python x_cot_h_m/ft_synt_stego.py \
  --model Qwen/Qwen3-0.6B \
  --train_jsonl x_cot_h_m/data/train.jsonl \
  --val_jsonl   x_cot_h_m/data/val.jsonl \
  --output_dir  x_cot_h_m/models/stego_lora_qwen06b_50k \
  --use_lora \
  --load_in_4bit \
  --bf16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --save_steps 500 --eval_steps 500 --logging_steps 10 \
  --save_total_limit 3 \
  --run_name stego_qwen06b_50k

max speed:
python x_cot_h_m/ft_synt_stego.py \
  --model Qwen/Qwen3-0.6B \
  --train_jsonl x_cot_h_m/data/train.jsonl \
  --val_jsonl   x_cot_h_m/data/val.jsonl \
  --output_dir  x_cot_h_m/models/stego_lora_qwen06b_50k \
  --use_lora \
  --bf16 \
  --use_flash_attention_2 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --save_total_limit 6 \
  --save_steps 1000 --eval_steps 1000 --logging_steps 10 \
  >> x_cot_h_m/models/stego_lora_qwen06b_50k/train.log 2>&1 &

python x_cot_h_m/ft_synt_stego.py \
  --model Qwen/Qwen3-0.6B \
  --train_jsonl x_cot_h_m/data/train.jsonl \
  --val_jsonl   x_cot_h_m/data/val.jsonl \
  --output_dir  x_cot_h_m/models/stego_lora_qwen06b_50k \
  --use_lora \
  --bf16 \
  --max_length 1536 \
  --learning_rate 2e-5 --weight_decay 0.0 --warmup_ratio 0.03 \
  --use_flash_attention_2 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1 \
  --save_total_limit 6 \
  --save_steps 1000 --eval_steps 1000 --logging_steps 10 \
  --answer_prefix_regex 'Answer\s*:\s*(?:-?\d+(?:\.\d+)?|unknown|[A-E])' \
  >> x_cot_h_m/models/stego_lora_qwen06b_50k/train.log 2>&1 &

python x_cot_h_m/ft_synt_stego.py \
  --model Qwen/Qwen3-0.6B \
  --train_jsonl x_cot_h_m/data/train.jsonl \
  --val_jsonl   x_cot_h_m/data/val.jsonl \
  --output_dir  x_cot_h_m/models/stego_lora_qwen06b_50k \
  --use_lora \
  --bf16 \
  --use_flash_attention_2 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --save_total_limit 6 \
  --save_steps 1000 --eval_steps 1000 --logging_steps 10 \
  >> x_cot_h_m/models/stego_lora_qwen06b_50k/train.log 2>&1 &
"""
import os, re, json, importlib.util, logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pathlib
import sys

if "MPLCONFIGDIR" not in os.environ:
    try:
        _mpl_dir = os.path.join(os.getcwd(), ".mplconfig")
        os.makedirs(_mpl_dir, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = _mpl_dir
    except Exception:
        pass

import torch
from torch.utils.data import Dataset

# Safe import of HF modules
def _maybe_stub_peft(use_lora: bool):
    if use_lora:
        return  # -> real peft

    import sys, types
    if "peft" in sys.modules:
        return  # already imported; assume OK

    peft_stub = types.ModuleType("peft")
    class PeftModel:
        pass
    peft_stub.PeftModel = PeftModel
    peft_stub.__all__ = ["PeftModel"]
    sys.modules["peft"] = peft_stub

def import_tf_modules(use_lora: bool):
    _maybe_stub_peft(use_lora)
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
        EarlyStoppingCallback,
        AutoModelForCausalLM,
    )
    import transformers
    return transformers, AutoConfig, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed, EarlyStoppingCallback, AutoModelForCausalLM

# Data utils
def setup_logger(log_level: str = "INFO"):
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line: 
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSON at line {i} in {path}: {e}")
    return rows

@dataclass
class DataRow:
    messages: List[Dict[str, str]]

class ChatJsonlDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, mask_mode: str = "cot_and_answer",
                 max_length: int = 2048, allow_truncation: bool = True,
                 answer_prefix: str = r"Answer\s*:\s*-?\d+(?:\.\d+)?",
                 supervise_think_inner: bool = True):
        self.rows = [DataRow(messages=row["messages"]) for row in rows]
        if not self.rows:
            raise ValueError("Empty dataset.")
        self.tok = tokenizer
        self.mask_mode = mask_mode
        self.max_length = max_length
        self.allow_truncation = allow_truncation
        self.answer_prefix = answer_prefix
        self.supervise_think_inner = supervise_think_inner
        if self.rows[0].messages[-1].get("role") != "assistant":
            raise ValueError("Each example must end with an 'assistant' message.")

    def __len__(self): return len(self.rows)

    def _render_prompt_and_target(self, messages: List[Dict[str, str]]):
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        if msgs[-1]["role"] != "assistant":
            raise ValueError("Last message must be 'assistant'.")
        # prefer chat template
        try:
            prompt_msgs = msgs[:-1] + [{"role": "assistant", "content": ""}]
            prompt_text = self.tok.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # fallback join
            parts = []
            for m in msgs[:-1]:
                parts.append(f"{m['role'].upper()}: {m['content']}")
            parts.append("ASSISTANT:")
            prompt_text = "\n".join(parts)
        assistant_text = msgs[-1]["content"]
        return prompt_text, assistant_text

    def _mask_labels(self, prompt_ids, full_ids, assistant_text: str):
        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100
        if self.mask_mode == "assistant":
            return labels
        asst_ids = self.tok(assistant_text, return_tensors="pt", add_special_tokens=False).input_ids
        start = prompt_ids.shape[1]
        end = start + asst_ids.shape[1]
        labels[:, start:end] = -100

        def token_span_from_char_span(c0, c1):
            prefix_ids = self.tok(assistant_text[:c0], return_tensors="pt", add_special_tokens=False).input_ids
            span_ids = self.tok(assistant_text[c0:c1], return_tensors="pt", add_special_tokens=False).input_ids
            return start + prefix_ids.shape[1], start + prefix_ids.shape[1] + span_ids.shape[1]

        spans = []
        if self.mask_mode in {"cot", "cot_and_answer"}:
            pat = r"<think>(.*?)</think>" if self.supervise_think_inner else r"(<think>.*?</think>)"
            m = re.search(pat, assistant_text, flags=re.DOTALL | re.IGNORECASE)
            if m:
                c0, c1 = (m.span(1))
                spans.append(token_span_from_char_span(c0, c1))
            else:
                # Think block missing in truncated assistant; warn once per process
                if not hasattr(self, "_warned_no_think"):
                    import logging
                    logging.warning("No <think>…</think> span found in assistant_text (possibly due to truncation).")
                    self._warned_no_think = True

        if self.mask_mode in {"answer_only", "cot_and_answer"}:
            # ---------- CHANGED: supervise EXACTLY 'Answer: <number>' ----------
            m = re.search(self.answer_prefix, assistant_text, flags=re.IGNORECASE)
            if m:
                c0, c1 = m.span(0)
                spans.append(token_span_from_char_span(c0, c1))
            else:
                # fallback to last non-empty line
                found_fallback = False
                for line in reversed(assistant_text.splitlines()):
                    if line.strip():
                        last = line
                        c0 = assistant_text.rfind(last); c1 = c0 + len(last)
                        spans.append(token_span_from_char_span(c0, c1))
                        found_fallback = True
                        break
                if not found_fallback and not hasattr(self, "_warned_no_answer"):
                    import logging
                    logging.warning("No Answer line matched and no fallback line found in assistant_text.")
                    self._warned_no_answer = True

        if not spans and self.mask_mode != "answer_only":
            spans.append((start, end))
        for t0, t1 in spans:
            labels[:, t0:t1] = full_ids[:, t0:t1]
        return labels

    def __getitem__(self, idx: int):
        row = self.rows[idx]

        prompt_text, assistant_text = self._render_prompt_and_target(row.messages)

        # tokenize prompt & assistant separately (no special tokens)
        prompt_enc = self.tok(prompt_text, return_tensors="pt", add_special_tokens=False)
        asst_enc   = self.tok(assistant_text, return_tensors="pt", add_special_tokens=False)

        prompt_ids = prompt_enc.input_ids
        asst_ids   = asst_enc.input_ids

        # KEEP THE WHOLE PROMPT, trim the ASSISTANT **TAIL** (i.e., keep the head) to fit max_length
        if prompt_ids.shape[1] >= self.max_length:
            raise ValueError(
                f"Prompt length {prompt_ids.shape[1]} >= max_length {self.max_length}. "
                "Increase --max_length or shorten this example."
            )

        room = self.max_length - prompt_ids.shape[1]
        # Keep the **first** `room` tokens of the assistant, drop overflow at the end
        asst_ids_trunc = asst_ids[:, :room] if asst_ids.shape[1] > room else asst_ids

        input_ids = torch.cat([prompt_ids, asst_ids_trunc], dim=1)
        attention_mask = torch.ones_like(input_ids)

        assistant_text_trunc = self.tok.decode(asst_ids_trunc[0], skip_special_tokens=True)

        # labels: mask prompt; selectively supervise CoT + Answer (tight span)
        labels = self._mask_labels(prompt_ids, input_ids, assistant_text_trunc)

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

# fallback loader
def _import_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    return module

def _normalize_automap_entry(entry):
    if isinstance(entry, str):
        return entry
    if isinstance(entry, list) and entry:
        return entry[0]
    if isinstance(entry, dict):
        for v in entry.values():
            if isinstance(v, str):
                return v
            if isinstance(v, list) and v:
                return v[0]
    raise ValueError(f"Unsupported auto_map entry format: {entry!r}")

def load_model_with_fallbacks(model_id: str, cache_dir: Optional[str], torch_dtype=None,
                              quantization_config=None, device_map="auto", AutoConfig=None, AutoModelForCausalLM=None):
    """
    initially- Try normal Auto* with trust_remote_code=True
    if it fails because arch is unknown to your Transformers, try to load via repo auto_map
    """
    import logging
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        hf_hub_download = None

    # Normal path
    try:
        cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, config=cfg, cache_dir=cache_dir, trust_remote_code=True,
            quantization_config=quantization_config, torch_dtype=torch_dtype, device_map=device_map,
        )
        return model
    except Exception as e:
        logging.warning(f"Auto load failed: {e}")

    # remote code via auto_map (only if can read config.json)
    if hf_hub_download is not None:
        cfg_path = hf_hub_download(repo_id=model_id, filename="config.json", cache_dir=cache_dir)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_json = json.load(f)
        auto_map = cfg_json.get("auto_map", {})
        if not auto_map:
            raise RuntimeError(
                "Could not load the model. Your Transformers version likely doesn't support this architecture, "
                "and the repo does not expose remote code via auto_map. Please upgrade transformers on your path "
                "(e.g., pip install -U transformers) or pick a model repo that provides remote code."
            )
        model_map = _normalize_automap_entry(auto_map.get("AutoModelForCausalLM"))
        config_map = _normalize_automap_entry(auto_map.get("AutoConfig"))
        model_py = model_map.split(".")[0] + ".py"
        config_py = config_map.split(".")[0] + ".py"
        model_file = hf_hub_download(repo_id=model_id, filename=model_py, cache_dir=cache_dir)
        config_file = hf_hub_download(repo_id=model_id, filename=config_py, cache_dir=cache_dir)
        model_module = _import_module_from_path("remote_modeling", model_file)
        config_module = _import_module_from_path("remote_configuration", config_file)
        ModelClass = getattr(model_module, model_map.split(".")[-1])
        ConfigClass = getattr(config_module, config_map.split(".")[-1])
        try:
            remote_cfg = ConfigClass.from_pretrained(model_id, cache_dir=cache_dir)
        except Exception:
            remote_cfg = ConfigClass(**cfg_json)
        model = ModelClass.from_pretrained(
            model_id, config=remote_cfg, cache_dir=cache_dir,
            quantization_config=quantization_config, torch_dtype=torch_dtype, device_map=device_map,
        )
        return model

    # No huggingface_hub available at all
    raise RuntimeError(
        "huggingface_hub is not available, and native Auto* load failed. "
        "Install it (pip install huggingface_hub) or upgrade transformers."
    )

# Main
def main():
    import argparse, os
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", default=None)
    parser.add_argument("--mask_mode", default="cot_and_answer", choices=["assistant","cot","cot_and_answer","answer_only"])
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)

    # model / training
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None)

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_flash_attention_2", action="store_true")

    # trainer args
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")

    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--save_strategy", default="steps", choices=["steps","epoch","no"])
    parser.add_argument("--eval_strategy", default="steps", choices=["no","steps","epoch"])

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--delete_optim_states_after_save", action="store_true")

    # early stopping
    parser.add_argument("--use_early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)

    # masking QoL
    parser.add_argument("--answer_prefix_regex", default=r"Answer\s*:\s*(?:-?\d+(?:\.\d+)?|unknown|[A-E])")
    parser.add_argument("--supervise_think_outer", action="store_true")
    parser.set_defaults(supervise_think_outer=True)

    # logging
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--log_level", default="INFO")

    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", default="eval_loss")
    parser.add_argument("--greater_is_better", action="store_true")

    args = parser.parse_args()
    setup_logger(args.log_level)

    print("argv:", sys.argv)
    print("CWD :", os.getcwd())
    print("TRAIN:", args.train_jsonl, "=>", pathlib.Path(args.train_jsonl).resolve())
    print("VAL  :", args.val_jsonl, "=>", pathlib.Path(args.val_jsonl).resolve())

    transformers, AutoConfig, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed, EarlyStoppingCallback, AutoModelForCausalLM = import_tf_modules(args.use_lora)
    logging.info(f"PyTorch version {torch.__version__} available.")

    set_seed(args.seed)

    # data
    rows_train = load_jsonl(args.train_jsonl)
    rows_val = load_jsonl(args.val_jsonl) if args.val_jsonl else None
    logging.info(f"Loaded {len(rows_train)} train; {0 if rows_val is None else len(rows_val)} val.")

    # tokeniser
    logging.info(f"Loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try: tok.padding_side = "right"
    except Exception: pass

    # quantisation
    quant_config = None
    if args.load_in_4bit or args.load_in_8bit:
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            )
        except Exception as e:
            raise RuntimeError("bitsandbytes is required for 4/8-bit. Install with `pip install bitsandbytes`.") from e

    # model
    logging.warning("If this is Qwen3 on an older Transformers, I'll try remote-code fallback next.")
    logging.info(f"Loading model: {args.model}")
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = load_model_with_fallbacks(
        model_id=args.model,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        device_map="auto",
        AutoConfig=AutoConfig,
        AutoModelForCausalLM=AutoModelForCausalLM,
    )

    # optional toggles
    if args.use_flash_attention_2:
        try:
            model.enable_input_require_grads()
            model.config.use_flash_attention_2 = True
            logging.info("Enabled FlashAttention 2.")
        except Exception as e:
            logging.warning(f"Failed to enable FlashAttention2: {e}")
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing.")
        except Exception as e:
            logging.warning(f"Failed to enable gradient checkpointing: {e}")

    # LoRA - if requested
    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        except Exception as e:
            raise RuntimeError("You requested --use_lora but `peft` import failed. "
                               "Install a Python-3.12-compatible peft or use a compatible Python.") from e
        if quant_config is not None:
            model = prepare_model_for_kbit_training(model)
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=target_modules, task_type=TaskType.CAUSAL_LM, bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        try: model.print_trainable_parameters()
        except Exception: pass

    # datasets
    ds_train = ChatJsonlDataset(
        rows_train, tok, mask_mode=args.mask_mode, max_length=args.max_length,
        answer_prefix=args.answer_prefix_regex, supervise_think_inner=not args.supervise_think_outer
    )
    ds_eval = ChatJsonlDataset(
        rows_val, tok, mask_mode=args.mask_mode, max_length=args.max_length,
        answer_prefix=args.answer_prefix_regex, supervise_think_inner=not args.supervise_think_outer
    ) if rows_val else None

    # collator
    def collate_fn(batch):
        input_ids = [b["input_ids"] for b in batch]
        attn = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tok.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # W&B
    if args.wandb_project:
        try:
            import wandb
            os.environ["WANDB_PROJECT"] = args.wandb_project
            if args.run_name: os.environ["WANDB_NAME"] = args.run_name
            wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        except Exception as e:
            logging.warning(f"W&B init failed: {e}")

    # evaluation strategy
    eval_strategy = args.eval_strategy
    if ds_eval is None and eval_strategy != "no":
        logging.warning("No validation set provided; switching eval_strategy to 'no'.")
        eval_strategy = "no"

    from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=eval_strategy,
        save_strategy=args.save_strategy,
        bf16=args.bf16, fp16=args.fp16,
        report_to=("wandb" if args.wandb_project else "none"),
        run_name=args.run_name,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
    )

    callbacks = []
    if args.use_early_stopping:
        if ds_eval is None or eval_strategy == "no":
            logging.warning("--use_early_stopping requires a validation set and eval strategy.")
        else:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold
            ))

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=collate_fn,
        tokenizer=tok,
        callbacks=callbacks,
    )

    try:
        n_all = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Trainable params: {n_train/1e6:.2f}M / {n_all/1e6:.2f}M")
    except Exception as _e:
        pass


    # delete optimiser/scheduler state after save to save disk
    if args.delete_optim_states_after_save:
        class _Cleaner:
            def __init__(self, out): self.out = out
            def on_save(self):
                from pathlib import Path
                import os, logging
                try:
                    for ckpt in sorted(Path(self.out).glob("checkpoint-*")):
                        for name in ("optimizer.pt","scheduler.pt"):
                            p = ckpt / name
                            if p.exists():
                                p.unlink()
                except Exception as e:
                    logging.warning(f"Failed to delete optimizer/scheduler states: {e}")
        cleaner = _Cleaner(args.output_dir)
        orig_save = trainer._save
        def _save_hook(output_dir=None, state_dict=None):
            out = orig_save(output_dir, state_dict); cleaner.on_save(); return out
        trainer._save = _save_hook  # type: ignore

    logging.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if ds_eval is not None:
        logging.info("Running evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    # write history CSV
    try:
        hist = trainer.state.log_history
        csv_path = os.path.join(args.output_dir, "loss_history.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("step,loss,eval_loss,learning_rate\n")
            for rec in hist:
                f.write(f"{rec.get('step','')},{rec.get('loss','')},{rec.get('eval_loss','')},{rec.get('learning_rate','')}\n")
        logging.info(f"Wrote {csv_path}")
    except Exception as e:
        logging.warning(f"Failed to write loss history: {e}")

    logging.info("Done.")

if __name__ == "__main__":
    main()
