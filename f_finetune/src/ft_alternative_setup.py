from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import logging
import math
import os
import random
import sys
import importlib
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# imports for alt paths
from datasets import load_from_disk  # only used if --use_pretokenized
import numpy as np  # used in p95 + optional pretokenized fallback

# env defaults & constants
os.environ.setdefault("TQDM_MININTERVAL", "60")
os.environ.setdefault("TQDM_MINITER", "200")
DEBUG_PROMPT_IDS = {1, 42, 321}  # helpful sanity-print ids


# Data containers
@dataclasses.dataclass
class Example:
    """lightish container used before tokenisation"""
    prompt_count: int
    instruction: str
    inp: str
    answer: str
    style: str
    dataset_source: str = "unknown"  # "first" or "second" or other

    def to_prompt(self, with_answer: bool = False, add_eos: bool = True) -> str:
        if self.inp:
            prompt = (
                f"### Instruction:\n{self.instruction}\n\n"
                f"### Input:\n{self.inp}\n\n"
                "### Response:\n"
            )
        else:
            prompt = f"### Instruction:\n{self.instruction}\n\n### Response:\n"
        if with_answer:
            prompt += self.answer
        if add_eos:
            prompt += tokenizer.eos_token
        return prompt


def build_chat_prompt(instruction: str, inp: str | None = "") -> str:
    """
    Return a single-turn chat prompt in the format Gemma-2-IT was trained on.
    We prepend a brief meta instruction inside the user message to encourage concise, on-point answers.
    add_generation_prompt=True inserts Gemma's assistant role marker!
    """
    meta = "Answer concisely and directly. Focus on task semantics; ignore stylistic tone cues. End after the answer."
    user_core = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    user_msg = f"{meta}\n\n{user_core}"
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def concise_generate(model, tokenizer, instruction: str, inp: str = "", max_new_tokens: int | None = None) -> str:
    """
    Deterministic, concise generation that mirrors training setup:
    - same chat template & brevity meta,
    - capped length,
    - stops on EOS or <end_of_turn>!
    """
    prompt_txt = build_chat_prompt(instruction, inp)
    enc = tokenizer(prompt_txt, return_tensors="pt").to(model.device)

    # Use p95 (computed during training) or the configured training cap if not provided
    gen_cap = max_new_tokens or (MAX_ANS_TOKENS if 'MAX_ANS_TOKENS' in globals() else 256)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=gen_cap,
            do_sample=False,           # deterministic & typically shorter
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.05,   # tiny nudge toward brevity
            eos_token_id=model.config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Strip the prompt portion from the decode
    gen = out[0, enc["input_ids"].shape[1]:]
    txt = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return txt


# Helper: tokenise a single Example -> dict(input_ids, labels)
def tokenise_example(ex: Example):
    # prompt part (masked out in labels)
    prompt_txt = build_chat_prompt(ex.instruction, ex.inp)
    prompt_ids = tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]

    # answer part (to be learned)
    ans_txt = ex.answer.strip()
    answer_ids = tokenizer(
        ans_txt, add_special_tokens=False, truncation=True, max_length=MAX_ANS_TOKENS
    )["input_ids"]
    # Exactly one end-of-turn token so the model learns to stop cleanly.
    answer_ids.append(EOT_ID)

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "labels": labels}


# Data-loading utilities

def three_way_split(
    ds: Dataset, *, val_pct: float = 0.05, test_pct: float = 0.05, seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """Group-wise split guaranteeing each *prompt_count* appears in one split"""
    rng = np.random.default_rng(seed)
    pcs = list({ex["prompt_count"] for ex in ds})
    rng.shuffle(pcs)
    n = len(pcs)
    n_val = int(n * val_pct)
    n_test = int(n * test_pct)
    val_ids = set(pcs[:n_val])
    test_ids = set(pcs[n_val: n_val + n_test])
    train_ids = set(pcs[n_val + n_test:])

    train = ds.filter(lambda ex: ex["prompt_count"] in train_ids)
    val = ds.filter(lambda ex: ex["prompt_count"] in val_ids)
    test = ds.filter(lambda ex: ex["prompt_count"] in test_ids)
    return train, val, test


def _p95_answer_length(tok, ds, sample_cap: int = 50000) -> int:
    """Compute 95th percentile of answer token lengths for a safe max_new_tokens cap"""
    n = min(len(ds), sample_cap)
    lens = []
    for i in range(n):
        a = ds[i].get("answer", "")
        if not a:
            continue
        lens.append(len(tok(a, add_special_tokens=False)["input_ids"]))
    return int(np.percentile(lens, 95)) if lens else 128


def load_style_list(paths: List[str] | None) -> Optional[set]:
    """Load a JSON list of style keys (e.g., instruct_*). Returns None if no paths provided"""
    if not paths:
        return None
    styles = set()
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                lst = json.load(fh)
            if isinstance(lst, list):
                styles.update(lst)
        except Exception as e:
            logging.warning("Could not read style list %s: %s", p, e)
    logging.info("Loaded %d styles from %d whitelist file(s).", len(styles), len(paths))
    return styles if styles else None


def _is_first_dataset_item(item: dict) -> bool:
    """Heuristic: FIRST dataset items have a 'paraphrases' list and 'instruction_original' at top level"""
    return isinstance(item, dict) and "paraphrases" in item and isinstance(item["paraphrases"], list)


def analyze_first_style_counts(paths: List[str], allowed_styles: Optional[set], require_content_score: int) -> set:
    """First pass over FIRST dataset(s) to count valid paraphrases per style; return styles >= min threshold (set later)"""
    from collections import Counter
    ctr = Counter()
    total_items = 0
    kept_items = 0
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for item in data:
            total_items += 1
            for ph in item.get("paraphrases", []):
                if ph.get("paraphrase_content_score", 0) < require_content_score:
                    continue
                sty = ph.get("instruct_type")
                if not sty:
                    continue
                if allowed_styles is not None and sty not in allowed_styles:
                    continue
                ctr[sty] += 1
                kept_items += 1
    logging.info("FIRST analysis: scanned %d items; candidate paraphrases: %d; distinct styles: %d",
                 total_items, kept_items, len(ctr))
    return set([s for s, c in ctr.items() if c >= FIRST_MIN_STYLE_COUNT])


def load_first_dataset_examples(
    paths: List[str],
    allowed_styles: set[str] | None,
    require_content_score: int = 5,
    min_style_count: int = 200,
    use_only_original_answer: bool = True,
) -> List[Example]:
    """
    Load FIRST dataset (500×326 structure with nested 'paraphrases'),
    keep only paraphrases with paraphrase_content_score==5 and allowed styles,
    and drop styles that don't have >= min_style_count such items globally!
    """
    # establish global valid styles (>= min_style_count)
    global FIRST_MIN_STYLE_COUNT
    FIRST_MIN_STYLE_COUNT = min_style_count
    valid_styles = analyze_first_style_counts(paths, allowed_styles, require_content_score)
    if allowed_styles is not None:
        valid_styles = valid_styles & allowed_styles  # intersection
    logging.info("FIRST valid styles after ≥%d filter: %d",
                 min_style_count, len(valid_styles))

    exs: List[Example] = []
    for path in paths:
        logging.info("Loading FIRST dataset file %s", path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for item in data:
            pc_id = item["prompt_count"]
            base_ans = item.get("output", "")
            inp = item.get("input", "") or item.get("scenarios", "") or ""

            # always include original
            exs.append(Example(pc_id, item["instruction_original"], inp, base_ans, "instruction_original", dataset_source="first"))

            for ph in item.get("paraphrases", []):
                if ph.get("paraphrase_content_score", 0) < require_content_score:
                    continue
                sty = ph.get("instruct_type")
                if not sty or (valid_styles and sty not in valid_styles):
                    continue
                parap = ph.get("paraphrase", "")
                if not parap:
                    continue
                ans = base_ans if use_only_original_answer else ph.get("answer", base_ans)
                exs.append(Example(pc_id, parap, inp, ans, sty, dataset_source="first"))

    random.shuffle(exs)
    logging.info("FIRST examples kept: %d", len(exs))
    return exs


def load_second_like_examples(paths: List[str], instruct_types: List[str], use_para_ans: bool) -> List[Example]:
    """Load SECOND-style prompts JSON(s): top-level instruct_* keys per item"""
    examples: List[Example] = []
    for p in paths:
        logging.info("Loading SECOND-like %s", p)
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for item in data:
            pc_id = item["prompt_count"]
            base_ans = item.get("output", "")
            inp = item.get("input", "")

            # always include the original instruction
            examples.append(
                Example(pc_id, item["instruction_original"], inp, base_ans, "instruction_original", dataset_source="second")
            )

            # Decide which paraphrase keys to keep
            if instruct_types:
                keep_keys = instruct_types
            else:  # default: keep all instruct_* keys found
                keep_keys = [k for k in item.keys() if k.startswith("instruct_")]

            for k in keep_keys:
                if k not in item:
                    continue
                paraphrase = item[k]
                if not paraphrase:
                    continue
                ans = item.get("output_paraphrase", base_ans) if use_para_ans else base_ans
                examples.append(Example(pc_id, paraphrase, inp, ans, k, dataset_source="second"))

                if pc_id in DEBUG_PROMPT_IDS:
                    logging.debug("[DBG %s-%s] %s", pc_id, k, paraphrase[:80])

    random.shuffle(examples)
    logging.info("SECOND-like examples kept: %d", len(examples))
    return examples


def load_examples_mixed(
    paths: List[str],
    instruct_types: List[str],
    use_para_ans: bool,
    style_whitelist: Optional[set],
    first_min_style_count: int,
    require_content_score: int = 5,
) -> List[Example]:
    """
    Load a mix of FIRST and SECOND files by auto-detecting the schema.
    Applies filtering for FIRST according to content score and style whitelist!
    """
    first_paths, second_paths = [], []
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data and isinstance(data, list) and _is_first_dataset_item(data[0]):
            first_paths.append(p)
        else:
            second_paths.append(p)

    examples: List[Example] = []
    if first_paths:
        logging.info("Detected FIRST-format files: %s", first_paths)
        examples.extend(
            load_first_dataset_examples(
                first_paths,
                allowed_styles=style_whitelist,
                require_content_score=require_content_score,
                min_style_count=first_min_style_count,
                use_only_original_answer=(not use_para_ans),
            )
        )
    if second_paths:
        logging.info("Detected SECOND-format files: %s", second_paths)
        # SECOND loader accepts arbitrary instruct_types; intersect if whitelist given
        keep_types = instruct_types
        if style_whitelist is not None:
            keep_types = [k for k in (keep_types or []) if k in style_whitelist] if keep_types else list(style_whitelist)
        examples.extend(load_second_like_examples(second_paths, keep_types, use_para_ans))

    random.shuffle(examples)
    logging.info("Total mixed examples: %d", len(examples))
    return examples


def load_examples_with_answers(
    prompts_path: str,
    answers_path: str,
    instruct_types: List[str],
    use_para_ans: bool,
) -> List[Example]:
    """
    Build examples where the *targets* come from the answers JSON.
    If use_para_ans=True, we use the paraphrase-specific answer when present,
    otherwise we fall back to answers['instruction_original'].
    For SECOND-like structure!
    """
    examples: List[Example] = []

    # read prompts
    with open(prompts_path, "r", encoding="utf-8") as fh:
        prompts_data = json.load(fh)

    # read answers -> map by prompt_count
    with open(answers_path, "r", encoding="utf-8") as fh:
        answers_data = json.load(fh)
    answers_map = {a["prompt_count"]: a for a in answers_data}

    for item in prompts_data:
        pc_id = item["prompt_count"]
        if pc_id not in answers_map:
            continue
        ans_rec = answers_map[pc_id]
        inp = item.get("input", "")

        # always include the original instruction
        orig_inst = item["instruction_original"]
        # target for original
        orig_ans = ans_rec.get("instruction_original", "")
        if orig_inst and orig_ans:
            examples.append(Example(pc_id, orig_inst, inp, orig_ans, "instruction_original", dataset_source="second"))

        # Decide which paraphrase keys to keep
        if instruct_types:
            keep_keys = [k for k in instruct_types if k in item]
        else:
            keep_keys = [k for k in item.keys() if k.startswith("instruct_")]

        for k in keep_keys:
            inst = item.get(k, "")
            if not inst:
                continue
            if use_para_ans:
                # prefer paraphrase-specific answer if present, else original
                ans = ans_rec.get(k, ans_rec.get("instruction_original", ""))
            else:
                ans = ans_rec.get("instruction_original", "")
            if ans:
                examples.append(Example(pc_id, inst, inp, ans, k, dataset_source="second"))

    random.shuffle(examples)
    return examples


# Collator that preserves metadata

class MetaDataCollator(DataCollatorForSeq2Seq):
    def __init__(self, *args, style2id: Optional[Dict[str, int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.style2id = style2id or {}

    def __call__(self, features):
        # Base tensors
        batch = super().__call__(features)
        # Pass through metadata where available
        if "prompt_count" in features[0]:
            batch["prompt_count"] = torch.tensor([int(f["prompt_count"]) for f in features], dtype=torch.long)
        if "style" in features[0]:
            style_ids = []
            for f in features:
                s = f.get("style", None)
                sid = self.style2id.get(s, -1) if s is not None else -1
                style_ids.append(sid)
            batch["style_id"] = torch.tensor(style_ids, dtype=torch.long)
        return batch


# LAP utilities

def _get_attr_chain(obj: Any, chain: str) -> Any:
    cur = obj
    for name in chain.split('.'):
        cur = getattr(cur, name)
    return cur


def resolve_transformer_layers(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    """
    Try common attribute paths across HF architectures to get the list/ModuleList of layers!
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    candidates = [
        "model.layers",
        "model.model.layers",
        "model.decoder.layers",        # T5/EncDec dec
        "transformer.h",               # GPT-2 style
        "gpt_neox.layers",             # GPT-NeoX
        "backbone.layers",
        "layers",                      # generic
    ]
    for path in candidates:
        try:
            layers = _get_attr_chain(base, path)
            if isinstance(layers, (list, torch.nn.ModuleList)):
                return layers
        except AttributeError:
            continue
    raise AttributeError("Could not locate transformer layers on the model; "
                         "please update resolve_transformer_layers() for your architecture.")


class LAPTtrainer(Trainer):
    """
    Trainer with Latent Adversarial Paraphrasing (LAP) training_step.
    (Only used when --use_lap is passed; default path uses vanilla Trainer.)
    Modified: we add a flag to disable extra losses during LAP inner compute_loss calls!
    """
    def __init__(self, *args, **kwargs):
        self.lap_kwargs = kwargs.pop('lap_kwargs', {})
        super().__init__(*args, **kwargs)
        self._in_lap_ce_only = False
        if self.lap_kwargs.get('use_lap', False):
            logging.info("LAP Trainer initialized with custom training step.")
            logging.info("LAP Config:\n" + json.dumps(self.lap_kwargs, indent=2))

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch: int = None) -> torch.Tensor:
        use_lap_for_batch = self.lap_kwargs.get('use_lap', False) and \
                            random.random() < self.lap_kwargs.get('lap_p_sample', 0.5)
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            if use_lap_for_batch:
                loss = self.lap_training_step(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)
        return loss.detach()

    def lap_training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Hyperparams
        t_inner     = int(self.lap_kwargs['lap_t_inner'])
        epsilon     = float(self.lap_kwargs['lap_epsilon'])
        delta_lr    = float(self.lap_kwargs['lap_delta_lr'])
        lambda_lr   = float(self.lap_kwargs['lap_lambda_lr'])
        layer_idx   = int(self.lap_kwargs['lap_layer'])

        # Temporarily disable gradient checkpointing
        ckpt_enabled = getattr(model, "is_gradient_checkpointing", False)
        if ckpt_enabled:
            model.gradient_checkpointing_disable()

        # Resolve target layer
        try:
            layers = resolve_transformer_layers(model)
            if not (0 <= layer_idx < len(layers)):
                raise IndexError(f"lap_layer={layer_idx} out of range [0, {len(layers)-1}]")
            target_layer = layers[layer_idx]
        except Exception as e:
            logging.error(f"[LAP] Failed to resolve target layer: {e}. Falling back to SFT.")
            if ckpt_enabled:
                model.gradient_checkpointing_enable()
            return self.compute_loss(model, inputs)

        # Probe to get the layer *input* tensor shape
        inp_buf = {}
        def size_probe(module, p_inputs, kwargs):
            inp_buf["h"] = p_inputs[0].detach()

        h_probe = target_layer.register_forward_pre_hook(size_probe, with_kwargs=True)
        with torch.no_grad():
            self._in_lap_ce_only = True
            out0 = model(**{k:v for k,v in inputs.items() if k in ["input_ids","attention_mask","labels"]})
            # mimic CE-only compute_loss to obtain baseline loss
            J0 = F.cross_entropy(
                out0.logits[..., :-1, :].contiguous().view(-1, out0.logits.size(-1)),
                inputs["labels"][..., 1:].contiguous().view(-1),
                ignore_index=-100
            ).detach()
            self._in_lap_ce_only = False
        h_probe.remove()

        if "h" not in inp_buf:
            logging.error("[LAP] Pre-hook size probe failed; falling back to SFT.")
            if ckpt_enabled:
                model.gradient_checkpointing_enable()
            return self.compute_loss(model, inputs)

        hidden_states_l = inp_buf["h"]
        delta = (1e-3 * torch.randn_like(hidden_states_l, dtype=torch.float32,
                                         device=hidden_states_l.device)).requires_grad_(True)
        log_lambda = torch.tensor(0.0, device=hidden_states_l.device, requires_grad=True)

        try:
            for i in range(t_inner):
                def add_delta_hook(module, p_inputs, kwargs):
                    d = delta
                    if d.dtype != p_inputs[0].dtype or d.device != p_inputs[0].device:
                        d = d.to(dtype=p_inputs[0].dtype, device=p_inputs[0].device)
                    return (p_inputs[0] + d,), kwargs

                h = target_layer.register_forward_pre_hook(add_delta_hook, with_kwargs=True)
                self._in_lap_ce_only = True
                out = model(**{k:v for k,v in inputs.items() if k in ["input_ids","attention_mask","labels"]})
                J_delta = F.cross_entropy(
                    out.logits[..., :-1, :].contiguous().view(-1, out.logits.size(-1)),
                    inputs["labels"][..., 1:].contiguous().view(-1),
                    ignore_index=-100
                )
                self._in_lap_ce_only = False
                h.remove()

                loss_constraint = (J_delta - J0).pow(2) - (epsilon ** 2)
                lagrangian = -delta.norm(p=2) + torch.exp(log_lambda).detach() * loss_constraint

                model.zero_grad(set_to_none=True)
                delta.grad = None
                lagrangian.backward()
                with torch.no_grad():
                    delta.add_(-delta_lr, delta.grad)
                delta.grad = None

                with torch.no_grad():
                    h = target_layer.register_forward_pre_hook(add_delta_hook, with_kwargs=True)
                    self._in_lap_ce_only = True
                    out2 = model(**{k:v for k,v in inputs.items() if k in ["input_ids","attention_mask","labels"]})
                    J_delta_updated = F.cross_entropy(
                        out2.logits[..., :-1, :].contiguous().view(-1, out2.logits.size(-1)),
                        inputs["labels"][..., 1:].contiguous().view(-1),
                        ignore_index=-100
                    )
                    self._in_lap_ce_only = False
                    h.remove()
                    violation = (J_delta_updated - J0).abs() - epsilon
                    log_lambda.add_(lambda_lr * torch.exp(log_lambda) * violation)

                del J_delta, J_delta_updated, lagrangian, loss_constraint, violation
                torch.cuda.empty_cache()

            final_delta = delta.detach()

        except Exception as e:
            logging.error(f"[LAP] Inner loop failed; falling back to SFT. Error: {e}", exc_info=True)
            if ckpt_enabled:
                model.gradient_checkpointing_enable()
            torch.cuda.empty_cache()
            return self.compute_loss(model, inputs)

        model.train()
        try:
            def add_final_delta_hook(module, p_inputs, kwargs):
                d = final_delta
                if d.dtype != p_inputs[0].dtype or d.device != p_inputs[0].device:
                    d = d.to(dtype=p_inputs[0].dtype, device=p_inputs[0].device)
                return (p_inputs[0] + d,), kwargs

            h = target_layer.register_forward_pre_hook(add_final_delta_hook, with_kwargs=True)
            self._in_lap_ce_only = True
            out3 = model(**{k:v for k,v in inputs.items() if k in ["input_ids","attention_mask","labels"]})
            outer_loss = F.cross_entropy(
                out3.logits[..., :-1, :].contiguous().view(-1, out3.logits.size(-1)),
                inputs["labels"][..., 1:].contiguous().view(-1),
                ignore_index=-100
            )
            self._in_lap_ce_only = False
        finally:
            if 'h' in locals():
                h.remove()
            if ckpt_enabled:
                model.gradient_checkpointing_enable()

        return outer_loss


# RobustTrainer: adds consistency KL, supervised contrastive, optional style-adversarial head

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class StyleAdvHead(nn.Module):
    def __init__(self, in_dim: int, num_styles: int, hidden: int = 0):
        super().__init__()
        layers = []
        if hidden and hidden > 0:
            layers = [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_styles)]
        else:
            layers = [nn.Linear(in_dim, num_styles)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RobustTrainer(LAPTtrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # running diagnostics
        self._extra_loss_sums = {"kld": 0.0, "contrast": 0.0, "style_adv": 0.0}
        self._extra_loss_counts = {"kld": 0, "contrast": 0, "style_adv": 0}
        self._warned_missing_fields = False

    def _log_extra(self, name: str, val: float):
        self._extra_loss_sums[name] += float(val)
        self._extra_loss_counts[name] += 1

    def compute_loss(self, model, inputs, return_outputs=False):
        # Filter inputs to model
        model_inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask", "labels"]}
        outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
        ce_loss = outputs.loss

        # During LAP inner loop, only return CE loss
        if getattr(self, "_in_lap_ce_only", False):
            return (ce_loss, outputs) if return_outputs else ce_loss

        loss = ce_loss
        B, T, V = outputs.logits.shape
        labels = inputs.get("labels")
        attention_mask = inputs.get("attention_mask")
        if labels is None or attention_mask is None:
            if not self._warned_missing_fields:
                logging.warning("Missing labels/attention_mask; skipping extra losses.")
                self._warned_missing_fields = True
            return (loss, outputs) if return_outputs else loss

        # Consistency KL over answer token distributions
        lam = getattr(self.args, "consistency_kld_weight", 0.0)
        if lam > 0 and "prompt_count" in inputs:
            with torch.no_grad():
                mask = (labels != -100)
            if mask.any():
                # mean distribution over answer tokens per example
                log_probs = F.log_softmax(outputs.logits[mask], dim=-1)   # [N, V]
                probs = log_probs.exp()
                # gather by example index
                ex_idx = torch.arange(B, device=labels.device).unsqueeze(1).expand(B, T)[mask]
                ex_sums = torch.zeros((B, V), device=probs.device, dtype=probs.dtype)
                counts = torch.zeros((B, 1), device=probs.device, dtype=probs.dtype)
                ex_sums.index_add_(0, ex_idx, probs)
                counts.index_add_(0, ex_idx, torch.ones_like(ex_idx, dtype=counts.dtype).unsqueeze(1))
                ex_means = ex_sums / counts.clamp_min(1.0)

                pcs = inputs["prompt_count"].tolist()
                # build one pair per pc if possible
                from collections import defaultdict
                by_pc = defaultdict(list)
                for i, pc in enumerate(pcs):
                    by_pc[pc].append(i)

                kld_total = outputs.logits.new_zeros(())
                pairs = 0
                for idxs in by_pc.values():
                    if len(idxs) < 2:  # need at least 2 paraphrases in the batch
                        continue
                    i, j = idxs[0], idxs[1]
                    p = ex_means[i].clamp_min(1e-8)
                    q = ex_means[j].clamp_min(1e-8)
                    kld_total = kld_total + 0.5 * (F.kl_div(p.log(), q, reduction="batchmean") + F.kl_div(q.log(), p, reduction="batchmean"))
                    pairs += 1
                if pairs > 0:
                    kld_term = kld_total / pairs
                    loss = loss + lam * kld_term
                    self._log_extra("kld", float(kld_term.detach().item()))
            else:
                pass  # no answer tokens found (shouldn't happen)

        # Supervised contrastive on prompt encodings
        cw = getattr(self.args, "contrast_weight", 0.0)
        if cw > 0 and "prompt_count" in inputs:
            hs_last = outputs.hidden_states[-1]   # [B, T, H]
            prompt_mask = (labels == -100) & (attention_mask == 1)
            pmask = prompt_mask.unsqueeze(-1)  # [B, T, 1]
            summed = (hs_last * pmask).sum(dim=1)   # [B, H]
            counts = pmask.sum(dim=1).clamp_min(1.0)
            pe = summed / counts  # [B, H]
            pe = F.normalize(pe, dim=-1)
            tau = getattr(self.args, "contrast_temperature", 0.1)
            sim = pe @ pe.t() / tau                   # [B, B]
            # mask out self
            self_mask = torch.eye(B, device=sim.device).bool()
            sim = sim.masked_fill(self_mask, float('-inf'))

            pcs = inputs["prompt_count"].tolist()
            pos_mask = torch.zeros_like(sim, dtype=torch.bool)
            for i in range(B):
                for j in range(B):
                    if i != j and pcs[i] == pcs[j]:
                        pos_mask[i, j] = True
            if pos_mask.any():
                log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
                pos_counts = pos_mask.sum(dim=1)
                safe = pos_counts > 0
                if safe.any():
                    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_counts.clamp_min(1)
                    contrast_loss = -pos_log_prob[safe].mean()
                    loss = loss + cw * contrast_loss
                    self._log_extra("contrast", float(contrast_loss.detach().item()))

        # Style adversarial head (optional)
        aw = getattr(self.args, "style_adv_weight", 0.0)
        if aw > 0 and "style_id" in inputs:
            hs_last = outputs.hidden_states[-1]   # [B, T, H]
            prompt_mask = (labels == -100) & (attention_mask == 1)
            pmask = prompt_mask.unsqueeze(-1)
            pe = (hs_last * pmask).sum(dim=1) / pmask.sum(dim=1).clamp_min(1.0)  # [B, H]

            # Build/attach head once
            if not hasattr(model, "style_adv_head"):
                in_dim = pe.size(-1)
                num_styles = int(inputs["style_id"].max().item() + 1)
                hidden = getattr(self.args, "style_adv_hidden", 0)
                model.style_adv_head = StyleAdvHead(in_dim, num_styles, hidden).to(pe.device)
                logging.info("Initialized StyleAdvHead(in=%d, num_styles=%d, hidden=%d)", in_dim, num_styles, hidden)

            # Gradient reversal
            lambd = getattr(self.args, "style_adv_lambda", 1.0)
            pe_rev = GradReverse.apply(pe, lambd)
            logits_style = model.style_adv_head(pe_rev)  # [B, S]
            # Only train on valid style ids >=0
            style_ids = inputs["style_id"]
            valid = style_ids >= 0
            if valid.any():
                style_loss = F.cross_entropy(logits_style[valid], style_ids[valid])
                loss = loss + aw * style_loss
                self._log_extra("style_adv", float(style_loss.detach().item()))

        return (loss, outputs) if return_outputs else loss


# CLI

def make_arg_parser():
    p = argparse.ArgumentParser(description="LoRA fine-tuning on CALIPER-like paraphrase data (robust/invariant extensions)")
    p.add_argument("--data_paths", nargs="+", required=True, help="One or more JSON files. FIRST or SECOND format auto-detected.")
    p.add_argument("--model_path", default="f_finetune/model")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", default="gemma_paraphrx_robust")
    p.add_argument(
        "--answers_path",
        default=None,
        help="Optional: path to answers JSON. If set, use answers[...] as targets instead of prompts[...]['output']. (SECOND-like only)",
    )

    # style selection & curriculum
    p.add_argument("--instruct_types", nargs="+", default=[],
                   help="Space-separated list of instruct_* keys to include; leave empty to use all paraphrases (SECOND only; FIRST ignores & uses whitelist).")
    p.add_argument("--style_whitelist_json", nargs="*", default=[],
                   help="JSON file(s) each containing a list of allowed instruct_* styles (XS/S/M/L/XL). If omitted, FIRST keeps styles by >=min count rule only; SECOND keeps all found.")
    p.add_argument("--first_min_style_count", type=int, default=200,
                   help="FIRST dataset: require at least this many paraphrases (content_score==5) per style.")
    p.add_argument("--require_content_score", type=int, default=5,
                   help="FIRST dataset: minimum paraphrase_content_score to keep (inclusive).")
    # target choice
    p.add_argument("--use_paraphrase_answer", action="store_true",
                   help="If set, use paraphrase-specific answers as targets when available (NOT recommended for invariance).")

    # splits
    p.add_argument("--val_pct", type=float, default=0.05)
    p.add_argument("--test_pct", type=float, default=0.05)

    # train hparams
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--max_answer_tokens", type=int, default=512,
                   help="Hard cap for target answer tokens. Used during tokenisation and as a sensible inference cap.")
    p.add_argument("--weight_decay", type=float, default=0.0)  # LoRA params only

    # LoRA / FT
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help='Comma-separated module names for LoRA. Use "all" for full fine-tune; use "none" for full fine-tune without LoRA.')
    p.add_argument("--lora_layer_idx", type=int, default=None,
                   help="If set, enable LoRA only on this transformer layer index (e.g., 6).")

    # infra
    p.add_argument("--use_deepspeed", action="store_true")
    p.add_argument("--deepspeed_config", default="ds_zero2.json")
    p.add_argument("--bnb_8bit_optim", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=5,
                   help="Keep at most N checkpoints. Use ≥2 when load_best_model_at_end=True to avoid missing best checkpoint.")
    p.add_argument("--model_only_checkpoints", action="store_true",
                   help="After each checkpoint save, delete optimizer/scheduler (and DeepSpeed zero) files to keep checkpoints small.")
    p.add_argument("--logging_steps", type=int, default=100, help="Log training loss every N steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_checkpoint", default=None,
                   help=("Path to a Trainer checkpoint directory to resume from "
                         "(e.g. checkpoint-6600). If omitted, training starts from scratch."))
    p.add_argument("--wandb_project", default="paraphrx_ft_50k")

    p.add_argument("--early_stopping_patience", type=int, default=4,
                   help="Number of evals with no improvement before stopping")

    # debug helpers
    p.add_argument("--debug_n_samples", type=int, default=0,
                   help="If >0, print N random tokenised samples and exit")
    p.add_argument("--debug_seed", type=int, default=123)

    # alternate pretokenized path
    p.add_argument("--use_pretokenized", action="store_true",
                   help="If set, load a preprocessed HF dataset (load_from_disk) instead of JSON + in-script tokenization.")
    p.add_argument("--tokenized_data_path", default=None,
                   help="Path passed to datasets.load_from_disk when --use_pretokenized is set.")

    # LAP toggles
    p.add_argument("--use_lap", action="store_true", help="Enable LAP training step (structure identical otherwise).")
    p.add_argument("--lap_layer", type=int, default=12)
    p.add_argument("--lap_t_inner", type=int, default=3)
    p.add_argument("--lap_p_sample", type=float, default=0.5)
    p.add_argument("--lap_epsilon", type=float, default=0.05)
    p.add_argument("--lap_delta_lr", type=float, default=1e-2)
    p.add_argument("--lap_lambda_lr", type=float, default=1e-3)

    # NEW: invariance regularizers
    p.add_argument("--consistency_kld_weight", type=float, default=0.0,
                   help="Weight for symmetrized KL consistency across paraphrases (0 disables).")
    p.add_argument("--contrast_weight", type=float, default=0.0,
                   help="Weight for supervised contrastive loss on prompt encodings (0 disables).")
    p.add_argument("--contrast_temperature", type=float, default=0.1,
                   help="Temperature for supervised contrastive loss.")
    p.add_argument("--style_adv_weight", type=float, default=0.0,
                   help="Weight for adversarial style confusion head (0 disables).")
    p.add_argument("--style_adv_hidden", type=int, default=0,
                   help="Hidden units in style adversarial head (0=linear).")
    p.add_argument("--style_adv_lambda", type=float, default=1.0,
                   help="Gradient reversal scaling lambda.")

    # curriculum (optional simple schedule)
    p.add_argument("--curriculum_style_jsons", nargs="*", default=[],
                   help="List of style-whitelist JSONs to apply per epoch stage (XS,S,M,L,XL etc.). If set, will swap styles across epochs.")
    p.add_argument("--curriculum_stage_epochs", nargs="*", type=int, default=[],
                   help="Epoch counts per curriculum stage (same length as curriculum_style_jsons).")

    return p


# Training summary / plots

def summarise_training(trainer: Trainer, out_dir: Path):
    hist = trainer.state.log_history
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics_history.json").write_text(json.dumps(hist, indent=2))

    # loss curves
    tr_steps, tr_loss, ev_steps, ev_loss = [], [], [], []
    for h in hist:
        if "loss" in h:
            tr_steps.append(h["step"])
            tr_loss.append(h["loss"])
        if "eval_loss" in h:
            ev_steps.append(h["step"])
            ev_loss.append(h["eval_loss"])
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        if tr_loss:
            plt.plot(tr_steps, tr_loss, label="train")
        if ev_loss:
            plt.plot(ev_steps, ev_loss, label="val")
        plt.legend()
        plt.grid(True)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png")
        plt.close()
    except Exception as e:
        logging.warning("Plot failed: %s", e)

    # simple text summary
    summary = {
        "train_examples": len(trainer.train_dataset) if trainer.train_dataset is not None else 0,
        "val_examples": len(trainer.eval_dataset) if trainer.eval_dataset else 0,
        "trainable_params": trainer.model.num_parameters(only_trainable=True),
        "total_params": trainer.model.num_parameters(),
    }
    if ev_loss:
        summary["final_val_loss"] = float(ev_loss[-1])
        summary["final_val_ppl"] = float(math.exp(ev_loss[-1]))
    with open(out_dir / "summary.txt", "w") as fh:
        for k, v in summary.items():
            fh.write(f"{k}: {v}\n")

    if wandb.run:  # log back
        wandb.save(str(out_dir / "summary.txt"))
        wandb.save(str(out_dir / "loss_curve.png"))
        wandb.save(str(out_dir / "metrics_history.json"))


# Checkpoint helpers
_CKPT_PATTERNS = (re.compile(r"checkpoint-(\d+)$"), re.compile(r"global_step(\d+)$"))


def _extract_step_from_name(name: str) -> int | None:
    for pat in _CKPT_PATTERNS:
        m = pat.search(name)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def find_latest_checkpoint_dir(out_dir: Path) -> Path | None:
    if not out_dir.exists():
        return None
    cands = []
    for d in out_dir.iterdir():
        if d.is_dir():
            step = _extract_step_from_name(d.name)
            if step is not None:
                cands.append((step, d))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    return cands[-1][1]


def _dir_size_gb(p: Path) -> float:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except Exception:
                pass
    return total / (1024 ** 3)


def prune_checkpoint_dir(ckpt_dir: Path):
    """Remove optimizer/scheduler and DeepSpeed ZeRO state files from a checkpoint dir"""
    if not ckpt_dir or not ckpt_dir.exists():
        return

    # Known single files (HF Trainer)
    remove_names = {
        "optimizer.pt",
        "optimizer.bin",
        "scheduler.pt",
        "scheduler.bin",
        "zero_pp_rank_0_mp_rank_00_optim_states.pt",
        "zero_pp_rank_0_mp_rank_00_optim_states.json",
    }

    # Remove known files at the root
    for name in remove_names:
        p = ckpt_dir / name
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    # Remove DS ZeRO optimizer shards and state dumps recursively
    for p in ckpt_dir.rglob("*"):
        name = p.name
        if not p.is_file():
            continue
        if (
            name.startswith("zero_") and "optim" in name
        ) or ("_optim_states" in name) or name.startswith("oslo_optim_states"):
            try:
                p.unlink()
            except Exception:
                pass

    # Log size after pruning
    logging.info("Pruned heavy files in %s | size now ≈ %.2f GB", ckpt_dir, _dir_size_gb(ckpt_dir))


def sanitize_trainer_state_best(ckpt_dir: Path):
    """Ensure best_model_checkpoint path points to an existing checkpoint to avoid dangling refs on resume"""
    try:
        st_path = ckpt_dir / "trainer_state.json"
        if not st_path.exists():
            return
        st = json.loads(st_path.read_text())
        best = st.get("best_model_checkpoint", "")
        if best and not Path(best).exists():
            st["best_model_checkpoint"] = str(ckpt_dir.resolve())
            st_path.write_text(json.dumps(st, indent=2))
            logging.info("Repointed best_model_checkpoint to %s", ckpt_dir)
    except Exception as e:
        logging.warning("sanitize_trainer_state_best failed: %s", e)


# Main

def main(argv=None):
    args = make_arg_parser().parse_args(argv)

    # Early W&B init
    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        job_type="finetune",
        config=vars(args),
    )

    # Logging setup
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{args.run_name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Starting run %s", args.run_name)
    logging.info("Command-line args:\n%s", json.dumps(vars(args), indent=2))

    # Seed & TF32
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Resolve dataset files / Artifacts
    dataset_paths = []
    hf_dataset_dir = None  # detect HF dataset directories passed via --data_paths

    for spec in args.data_paths:
        p = Path(spec)

        if p.is_dir() and (p / "dataset_dict.json").exists():
            if hf_dataset_dir is None:
                hf_dataset_dir = str(p.resolve())
                logging.info("Detected HuggingFace dataset directory in --data_paths: %s", hf_dataset_dir)
            else:
                logging.warning("Additional HF dataset directory ignored (already set to %s): %s", hf_dataset_dir, str(p))
            continue

        if p.exists():
            dataset_paths.append(str(p))
        else:
            logging.info("Downloading Artifact %s", spec)
            art = run.use_artifact(f"{args.wandb_project}/{spec}:latest", type="dataset")
            ddir = Path(art.download())
            files = list(ddir.glob("*.json"))
            if not files:
                raise FileNotFoundError(f"No JSON in artifact {spec} (downloaded into {str(ddir)})")
            dataset_paths.append(str(files[0]))

    if hf_dataset_dir and not args.use_pretokenized:
        args.use_pretokenized = True
        args.tokenized_data_path = hf_dataset_dir
        logging.info("Auto-enabled --use_pretokenized with --tokenized_data_path=%s", hf_dataset_dir)

    logging.info("Datasets: %s", dataset_paths if dataset_paths else ["<none (using pretokenized)>"])

    # Tokeniser & model
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def _get_eot_id(tok):
        tid = tok.convert_tokens_to_ids("<end_of_turn>")
        return tid if tid is not None and tid != tok.unk_token_id else tok.eos_token_id

    # Global EOT used during tokenisation
    global EOT_ID
    EOT_ID = _get_eot_id(tokenizer)
    global MAX_ANS_TOKENS
    MAX_ANS_TOKENS = args.max_answer_tokens

    # Decide LoRA vs full FT robustly (treat "none" as full FT to avoid PEFT crash)
    tm = (args.target_modules or "").strip()
    tm_lower = tm.lower()
    if tm_lower in {"all", "none"}:
        full_ft = True
        if tm_lower == "none":
            logging.info("--target_modules none -> running full-parameter fine-tune (no LoRA).")
        else:
            logging.info("--target_modules all -> running full-parameter fine-tune (no LoRA).")
    else:
        full_ft = False

    if full_ft:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
    else:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            offload_state_dict=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map="auto", quantization_config=bnb_cfg
        )
        model = prepare_model_for_kbit_training(model)

    if not full_ft:
        mods = [m.strip() for m in tm.split(",") if m.strip()]
        if not mods:
            raise ValueError("No valid --target_modules provided for LoRA. Pass a comma-separated list or use 'none'/'all'.")
        lcfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=mods,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lcfg)
        logging.info("LoRA on modules: %s", mods)

        if args.lora_layer_idx is not None:
            needle = f".layers.{args.lora_layer_idx}."
            kept, frozen = 0, 0
            for n, p in model.named_parameters():
                if "lora_" in n:
                    if needle in n:
                        p.requires_grad_(True); kept += p.numel()
                    else:
                        p.requires_grad_(False); frozen += p.numel()
            logging.info("LoRA single-layer mode: kept %d params in %s; froze %d elsewhere",
                         kept, needle, frozen)
    else:
        logging.info("Full-parameter fine-tune (no LoRA)")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # needed for checkpointing
    model.config.eos_token_id = list({tokenizer.eos_token_id, EOT_ID})

    # DATA PATHS
    using_pretok = bool(args.use_pretokenized and args.tokenized_data_path)
    style_whitelist = load_style_list(args.style_whitelist_json)

    if not using_pretok:
        # Build examples
        if args.answers_path:
            if len(dataset_paths) != 1:
                raise ValueError("When --answers_path is provided, pass exactly one prompts JSON in --data_paths.")
            examples = load_examples_with_answers(
                prompts_path=dataset_paths[0],
                answers_path=args.answers_path,
                instruct_types=args.instruct_types,
                use_para_ans=args.use_paraphrase_answer,
            )
        else:
            examples = load_examples_mixed(
                dataset_paths,
                instruct_types=args.instruct_types,
                use_para_ans=args.use_paraphrase_answer,
                style_whitelist=style_whitelist,
                first_min_style_count=args.first_min_style_count,
                require_content_score=args.require_content_score,
            )

        if args.debug_n_samples > 0:
            random.seed(args.debug_seed)
            for ex in random.sample(examples, min(len(examples), args.debug_n_samples)):
                toks = tokenise_example(ex)
                logging.info(
                    "DEBUG SAMPLE:\nPROMPT:\n%s\nANSWER:\n%s\nTOKENS:%s\nLABELS:%s",
                    ex.to_prompt(False),
                    ex.answer,
                    toks["input_ids"][:40],
                    toks["labels"][:40],
                )
            logging.info("Debug sample print complete. Exiting by request (--debug_n_samples>0).")
            return

        raw_ds = Dataset.from_list([dataclasses.asdict(e) for e in examples])
        # Keep a copy of full raw for potential curriculum
        full_raw_ds = raw_ds

        # Initial style set for training (curriculum stage 0)
        if args.curriculum_style_jsons:
            stage_styles = load_style_list([args.curriculum_style_jsons[0]])
            if stage_styles:
                raw_ds = raw_ds.filter(lambda ex: (ex["style"] in stage_styles) or (ex["style"] == "instruction_original"))
                logging.info("Applied curriculum stage-0 style filter: %d examples remain.", len(raw_ds))

        train_raw, val_raw, test_raw = three_way_split(raw_ds, val_pct=args.val_pct, test_pct=args.test_pct, seed=args.seed)
        logging.info("raw sizes – train: %d | val: %d | test: %d", len(train_raw), len(val_raw), len(test_raw))

        # Recommend a concise generation cap for later inference
        try:
            p95_len = _p95_answer_length(tokenizer, full_raw_ds)
            logging.info("Answer length p95 (tokens): %d — use as max_new_tokens for inference.", p95_len)
            with open(Path(args.output_dir) / "p95_answer_tokens.txt", "w") as fh:
                fh.write(str(p95_len) + "\n")
        except Exception as _e:
            logging.warning("Could not compute p95 answer length: %s", _e)

        def batch_tokenise(batch):
            iids, lbls, styles, pcs = [], [], [], []
            for pc, ins, inp, ans, sty in zip(batch["prompt_count"], batch["instruction"], batch["inp"], batch["answer"], batch["style"]):
                tok = tokenise_example(Example(pc, ins, inp, ans, sty))
                iids.append(tok["input_ids"])
                lbls.append(tok["labels"])
                styles.append(sty)
                pcs.append(pc)
            return {"input_ids": iids, "labels": lbls, "style": styles, "prompt_count": pcs}

        train_ds = train_raw.map(batch_tokenise, batched=True, remove_columns=train_raw.column_names)
        val_ds = val_raw.map(batch_tokenise, batched=True, remove_columns=val_raw.column_names)
        test_ds = test_raw.map(batch_tokenise, batched=True, remove_columns=test_raw.column_names)
        logging.info("tokenised – train: %d | val: %d | test: %d", len(train_ds), len(val_ds), len(test_ds))

        # Build style2id from training data (for style adversarial head & logging)
        all_styles = list(sorted(set(train_ds["style"])))
        style2id = {s: i for i, s in enumerate(all_styles)}
        logging.info("Training styles: %d (example: %s)", len(style2id), list(style2id)[:8])

    else:
        logging.info(f"Loading pre-tokenized data from {args.tokenized_data_path}")
        tokenized = load_from_disk(args.tokenized_data_path)
        train_ds = tokenized["train"]
        val_ds = tokenized.get("validation", None) or tokenized.get("val", None)
        test_ds = tokenized.get("test", None)
        if val_ds is None:
            logging.warning("No 'validation' split found; using a small slice of train for eval.")
            val_ds = train_ds.select(range(min(len(train_ds), 1024)))
        logging.info("tokenised – train: %d | val: %d | test: %d", len(train_ds), len(val_ds), (len(test_ds) if test_ds else 0))
        style2id = {}  # unknown; extra losses may be disabled if missing metadata

    # Collator (pass through metadata)
    collator = MetaDataCollator(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8, style2id=style2id
    )

    # DeepSpeed option
    ds_cfg = args.deepspeed_config if args.use_deepspeed else None
    if args.use_deepspeed:
        try:
            importlib.import_module("deepspeed")
        except ImportError:
            logging.warning("DeepSpeed not available, disabling")
            ds_cfg = None

    targs = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.num_epochs,
        optim=("adamw_bnb_8bit" if args.bnb_8bit_optim else "adamw_torch"),
        bf16=args.bf16,
        fp16=not (args.bf16 or args.bnb_8bit_optim),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        group_by_length=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=0.3,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=["wandb"],
        deepspeed=ds_cfg,
        seed=args.seed,
    )

    class StepDigest(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kwargs):
            if not logs:
                return

            # print train loss every logging_steps
            if "loss" in logs and state.global_step % args_.logging_steps == 0:
                # extra loss moving avgs if available
                if isinstance(kwargs.get("model"), AutoModelForCausalLM):
                    pass
                logging.info(
                    "step %5d | train_loss %.4f | lr %.3g",
                    state.global_step,
                    logs["loss"],
                    logs.get("learning_rate", float("nan")),
                )
                # also dump running averages of extra losses if RobustTrainer
                tr = kwargs.get("trainer", None)
                if tr and hasattr(tr, "_extra_loss_sums"):
                    ex = tr._extra_loss_sums; ct = tr._extra_loss_counts
                    kld = ex["kld"]/max(1, ct["kld"]); con = ex["contrast"]/max(1, ct["contrast"]); adv = ex["style_adv"]/max(1, ct["style_adv"])
                    logging.info("   extras | kld~%.4f  contrast~%.4f  style_adv~%.4f", kld, con, adv)
                    if wandb.run:
                        wandb.log({"loss_consistency_kld_avg": kld, "loss_contrast_avg": con, "loss_style_adv_avg": adv}, step=state.global_step)

            if "eval_loss" in logs:
                logging.info("step %5d | eval_loss  %.4f | perplexity %.2f",
                             state.global_step, logs["eval_loss"], math.exp(logs["eval_loss"]))

    class ModelOnlyCheckpointCallback(TrainerCallback):
        def on_save(self, args_, state, control, **kwargs):
            try:
                ckpt_dir = find_latest_checkpoint_dir(Path(args_.output_dir))
                if ckpt_dir is None:
                    return
                prune_checkpoint_dir(ckpt_dir)
            except Exception as e:
                logging.warning("Prune failed: %s", e)

    if full_ft:
        for p_ in model.parameters():
            p_.requires_grad_(True)

    # Trainer class: robust
    trainer_cls = RobustTrainer
    trainer_kwargs = dict(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=(
            [StepDigest(), EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
            + ([ModelOnlyCheckpointCallback()] if args.model_only_checkpoints else [])
        ),
    )
    if args.use_lap:
        lap_cfg = {k: getattr(args, k) for k in [
            "use_lap", "lap_layer", "lap_t_inner", "lap_p_sample", "lap_epsilon", "lap_delta_lr", "lap_lambda_lr"
        ]}
        trainer_kwargs["lap_kwargs"] = lap_cfg

    trainer = trainer_cls(**trainer_kwargs)

    # Optional: curriculum swapping per epoch
    if args.curriculum_style_jsons and args.curriculum_stage_epochs:
        class CurriculumCallback(TrainerCallback):
            def __init__(self, trainer_ref, full_raw: Dataset, stage_files: List[str], stage_epochs: List[int]):
                self.trainer_ref = trainer_ref
                self.full_raw = full_raw
                self.stage_files = stage_files
                self.stage_epochs = stage_epochs
                self.stage_idx = 0
                self.stage_start_epoch = 0

            def on_epoch_begin(self, args_, state, control, **kwargs):
                # Determine current stage by completed epochs
                ep = int(state.epoch) if state.epoch is not None else 0
                # move to next stage if needed
                cum = 0
                new_stage = 0
                for i, ecount in enumerate(self.stage_epochs):
                    cum += ecount
                    if ep < cum:
                        new_stage = i
                        break
                if new_stage != self.stage_idx:
                    self.stage_idx = new_stage
                    styles = load_style_list([self.stage_files[self.stage_idx]])
                    if styles:
                        # rebuild train dataset
                        filtered = self.full_raw.filter(lambda ex: (ex["style"] in styles) or (ex["style"] == "instruction_original"))
                        logging.info("[Curriculum] Stage %d -> styles=%d | examples=%d", self.stage_idx, len(styles), len(filtered))
                        def batch_tokenise2(batch):
                            iids, lbls, styles_, pcs = [], [], [], []
                            for pc, ins, inp, ans, sty in zip(batch["prompt_count"], batch["instruction"], batch["inp"], batch["answer"], batch["style"]):
                                tok = tokenise_example(Example(pc, ins, inp, ans, sty))
                                iids.append(tok["input_ids"]); lbls.append(tok["labels"]); styles_.append(sty); pcs.append(pc)
                            return {"input_ids": iids, "labels": lbls, "style": styles_, "prompt_count": pcs}
                        new_train = filtered.map(batch_tokenise2, batched=True, remove_columns=filtered.column_names)
                        self.trainer_ref.train_dataset = new_train
                        # rebuild style2id for collator
                        all_styles2 = list(sorted(set(new_train["style"])))
                        self.trainer_ref.data_collator.style2id = {s:i for i,s in enumerate(all_styles2)}

        # Attach callback only when not using pretokenized (we need raw)
        if not using_pretok:
            trainer.add_callback(CurriculumCallback(trainer, full_raw_ds, args.curriculum_style_jsons, args.curriculum_stage_epochs))
            logging.info("Curriculum callback attached with %d stages.", len(args.curriculum_style_jsons))

    logging.info("Step-0 evaluation (no fine-tuning)")
    init_metrics = trainer.evaluate()
    logging.info("step 0 | eval_loss %.4f | ppl %.2f", init_metrics["eval_loss"], math.exp(init_metrics["eval_loss"]))
    if wandb.run:
        wandb.log({"eval_loss/step0": init_metrics["eval_loss"]})

    # Train (with optional LR catch-up if resuming from a pruned checkpoint)
    if args.resume_checkpoint:
        ck = Path(args.resume_checkpoint)
        try:
            st_path = ck / "trainer_state.json"
            if st_path.exists():
                st = json.loads(st_path.read_text())
                best = st.get("best_model_checkpoint", "")
                if best and not Path(best).exists():
                    st["best_model_checkpoint"] = str(ck.resolve())
                    st_path.write_text(json.dumps(st, indent=2))
                    logging.info("Repointed best_model_checkpoint to %s", ck)
        except Exception as e:
            logging.warning("sanitize_trainer_state_best failed: %s", e)

        opt_file = ck / "optimizer.pt"
        sched_file = ck / "scheduler.pt"
        if (not opt_file.exists()) and (not sched_file.exists()):
            try:
                st = json.loads((ck / "trainer_state.json").read_text())
                gs = int(st.get("global_step", 0))
                if gs > 0 and hasattr(trainer, "lr_scheduler") and trainer.lr_scheduler is not None:
                    for _ in range(gs):
                        trainer.lr_scheduler.step()
                    logging.info("Advanced LR scheduler by %d steps to match resumed global_step.", gs)
            except Exception as e:
                logging.warning("Failed LR scheduler catch-up: %s", e)

    trainer.train(resume_from_checkpoint=args.resume_checkpoint)

    logging.info("Total training steps: %d", trainer.state.max_steps)
    # Save final (adapters if PEFT, full if full_ft)
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logging.info("Training done: model & tokenizer in %s", final_dir)

    # style-wise validation loss
    from collections import defaultdict
    val_loader = torch.utils.data.DataLoader(
        val_ds.remove_columns(["style"]) if "style" in val_ds.column_names else val_ds,  # model feed
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
    )
    styles = val_ds["style"] if "style" in val_ds.column_names else ["_all"] * len(val_ds)  # parallel list

    model.eval()
    losses_by_style = defaultdict(list)
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            model_inputs = {
                k: v.to(model.device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask", "labels"]
            }
            out = model(**model_inputs)
            bsz = batch["input_ids"].size(0)
            sl = styles[idx * bsz: (idx + 1) * bsz]
            for s in sl:
                losses_by_style[s].append(out.loss.item())

    style_avg = {k: float(np.mean(v)) for k, v in losses_by_style.items()}
    for sty, lv in sorted(style_avg.items(), key=lambda x: x[1], reverse=True):
        logging.info("VAL-loss %-25s %.4f", sty, lv)

    if wandb.run:
        wandb.log({f"val_loss/{sty}": lv for sty, lv in style_avg.items()})

    summarise_training(trainer, Path(args.output_dir))

    # optional: evaluate on held-out test set
    if 'test_ds' in locals() and test_ds is not None:
        test_metrics = trainer.evaluate(test_ds)
        logging.info("TEST loss %.4f | ppl %.2f", test_metrics["eval_loss"], math.exp(test_metrics["eval_loss"]))

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
