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
from typing import List, Tuple, Dict, Any, Iterable

import torch
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

# Environment defaults & constants
os.environ.setdefault("TQDM_MININTERVAL", "60")
os.environ.setdefault("TQDM_MINITER", "200")
DEBUG_PROMPT_IDS = {1, 42, 321}  # helpful sanity‑print ids


@dataclasses.dataclass
class Example:
    """lightish container used before tokenisation"""
    prompt_count: int
    instruction: str
    inp: str
    answer: str
    style: str

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
    add_generation_prompt=True inserts Gemma's assistant role marker.
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
    - stops on EOS or <end_of_turn>.
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

    # answer part (to be learned) — ALWAYS the 'output' field (canonical)
    ans_txt = ex.answer.strip()
    answer_ids = tokenizer(
        ans_txt, add_special_tokens=False, truncation=True, max_length=MAX_ANS_TOKENS
    )["input_ids"]
    # Exactly one end-of-turn token so the model learns to stop cleanly.
    answer_ids.append(EOT_ID)

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "labels": labels}


# Data‑loading utilities
def three_way_split(
    ds: Dataset, *, val_pct: float = 0.05, test_pct: float = 0.05, seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """Group‑wise split guaranteeing each *prompt_count* appears in one split"""

    import numpy as np

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
    import numpy as np

    n = min(len(ds), sample_cap)
    lens = []
    for i in range(n):
        a = ds[i].get("answer", "")
        if not a:
            continue
        lens.append(len(tok(a, add_special_tokens=False)["input_ids"]))
    return int(np.percentile(lens, 95)) if lens else 128


def _load_styles_list(path: str | None) -> List[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as fh:
        lst = json.load(fh)
    if not isinstance(lst, list):
        raise ValueError(f"--styles_paths file must be a JSON array of strings: {path}")
    # normalize/strip
    return [str(x).strip() for x in lst if str(x).strip()]


def load_examples_multi(paths: List[str], styles_per_path: List[List[str]] | None, use_para_ans: bool) -> List[Example]:
    """
    Load multiple flattened prompt JSONs (e.g., SECOND + FIRST-instructions-only),
    each with its own selection of instruct_* keys given in styles_per_path[i].
    Always includes the instruction_original.
    ALWAYS uses 'output' as the target answer (canonicalization).
    """
    examples: List[Example] = []

    for idx, p in enumerate(paths):
        logging.info("Loading %s", p)
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        keep_keys = None
        if styles_per_path and idx < len(styles_per_path) and styles_per_path[idx]:
            keep_keys = set(styles_per_path[idx])  # dataset-specific styles
            if "instruction_original" in keep_keys:
                keep_keys.discard("instruction_original")  # included anyway

        for item in data:
            pc_id = item["prompt_count"]
            base_ans = item.get("output", "")  # canonical target
            inp = item.get("input", "")

            # Always include the original instruction
            if "instruction_original" in item and base_ans:
                examples.append(
                    Example(pc_id, item["instruction_original"], inp, base_ans, "instruction_original")
                )

            # Select paraphrase keys
            if keep_keys is None:
                all_keys = [k for k in item.keys() if k.startswith("instruct_")]
            else:
                all_keys = [k for k in keep_keys if k in item]

            for k in all_keys:
                paraphrase = item.get(k, "")
                if not paraphrase:
                    continue
                # We *always* predict the canonical 'output' (ignore paraphrase-specific answers)
                ans = base_ans
                examples.append(Example(pc_id, paraphrase, inp, ans, k))

                if pc_id in DEBUG_PROMPT_IDS:
                    logging.debug("[DBG %s-%s] %s", pc_id, k, paraphrase[:120])

    random.shuffle(examples)
    return examples


# Optional LAP augmentation helpers (layer resolution etc.)
def _get_attr_chain(obj: Any, chain: str) -> Any:
    cur = obj
    for name in chain.split('.'):
        cur = getattr(cur, name)
    return cur


def resolve_transformer_layers(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    """
    Try common attribute paths across HF architectures to get the list/ModuleList of layers.
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


class ParaphrxTrainer(Trainer):
    """
    Trainer extended with:
      - Optional LAP adversarial step (when --use_lap is set)
      - Optional consistency regularizer across paraphrases of the same prompt_count
      - Optional GroupDRO weighting by 'style'
      - Optional ORPO stage (preference optimization)
    """
    def __init__(self, *args, **kwargs):
        self.lap_kwargs = kwargs.pop('lap_kwargs', {})
        super().__init__(*args, **kwargs)
        if self.lap_kwargs.get('use_lap', False):
            logging.info("LAP Trainer path enabled.")
            logging.info("LAP Config:\n" + json.dumps(self.lap_kwargs, indent=2))

        # runtime state for GroupDRO
        self.group_w: Dict[str, float] = {}

    # ORPO helpers
    def _nll_per_sample(self, logits, labels) -> torch.Tensor:
        """
        Compute negative log-likelihood per sample, summing over non-ignored tokens.
        logits: (B, T, V); labels: (B, T) with -100 masked
        returns: (B,) NLL
        """
        logp = torch.nn.functional.log_softmax(logits, dim=-1)  # (B,T,V)
        # For gather, set all ignore positions to 0 index safely
        targ = labels.clone()
        targ[targ == -100] = 0
        token_ll = torch.gather(logp, dim=-1, index=targ.unsqueeze(-1)).squeeze(-1)  # (B,T)
        mask = (labels != -100)
        token_ll = token_ll * mask  # zero out ignored
        seq_ll = token_ll.sum(dim=1)  # (B,)
        return -seq_ll  # NLL per sample

    def _orpo_loss(self, model, inputs):
        """
        ORPO loss over a batch of pairs.
        Expect keys: input_ids_chosen, labels_chosen, input_ids_rejected, labels_rejected, weight(optional)
        """
        # Extract and move tensors; drop non-model kwargs
        keys_model = ["input_ids_chosen", "labels_chosen", "input_ids_rejected", "labels_rejected", "attention_mask_chosen", "attention_mask_rejected"]
        model_kwargs = {k: v for k, v in inputs.items() if k in keys_model}
        device = next(model.parameters()).device
        for k, v in list(model_kwargs.items()):
            model_kwargs[k] = v.to(device)

        out_c = model(input_ids=model_kwargs["input_ids_chosen"], labels=model_kwargs["labels_chosen"])
        out_r = model(input_ids=model_kwargs["input_ids_rejected"], labels=model_kwargs["labels_rejected"])
        nll_c = self._nll_per_sample(out_c.logits, model_kwargs["labels_chosen"])
        nll_r = self._nll_per_sample(out_r.logits, model_kwargs["labels_rejected"])

        beta = getattr(self.args, "orpo_beta", 0.1)
        margin = beta * (-nll_c + nll_r)  # larger is better
        loss_vec = torch.nn.functional.softplus(-margin)  # -log σ(margin)

        if "weight" in inputs:
            w = inputs["weight"].to(loss_vec.device, dtype=loss_vec.dtype)
            loss_vec = loss_vec * w

        return loss_vec.mean()

    # Consistency + GroupDRO (SFT path)
    def _compute_sft_loss_with_regularizers(self, model, inputs):
        """
        Compute token-level CE per sample for SFT, optionally apply:
         - GroupDRO weighting by style
         - Consistency regularizer across paraphrases with same prompt_count
        """
        # Separate model inputs and meta
        meta = {}
        for k in ["style", "prompt_count", "weight"]:
            if k in inputs:
                meta[k] = inputs.pop(k)

        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        labels = inputs["labels"]
        # per-sample NLL
        loss_vec = self._nll_per_sample(logits, labels)  # (B,)

        # GroupDRO weighting
        if getattr(self.args, "use_groupdro", False) and "style" in meta:
            styles = meta["style"]
            eta = getattr(self.args, "groupdro_eta", 0.05)
            # update group weights by current group average loss
            sums, counts = {}, {}
            # collect on CPU for safety
            lv = loss_vec.detach().cpu().tolist()
            for l, s in zip(lv, styles):
                sums[s] = sums.get(s, 0.0) + l
                counts[s] = counts.get(s, 0) + 1
                if s not in self.group_w:
                    self.group_w[s] = 0.0  # log weight
            for s in counts:
                g_loss = sums[s] / max(1, counts[s])
                self.group_w[s] += eta * g_loss
            # softmax-normalized positive weights
            ws = {s: math.exp(w) for s, w in self.group_w.items()}
            Z = sum(ws.values()) or 1.0
            ws = {s: v / Z for s, v in ws.items()}
            w_vec = torch.tensor([ws.get(s, 1.0 / max(1, len(ws))) for s in styles],
                                 device=loss_vec.device, dtype=loss_vec.dtype)
            ce = (loss_vec * w_vec).mean()
        else:
            ce = loss_vec.mean()

        # Consistency regularizer (paraphrase invariance)
        lam = getattr(self.args, "consistency_lambda", 0.0)
        cons_loss = 0.0
        if lam > 0 and "prompt_count" in meta:
            # Pool representations at last non-pad token
            hs = outputs.hidden_states[-1]  # (B,T,H)
            attn = inputs.get("attention_mask", None)
            if attn is None:
                attn = (labels != -100).to(hs.device, dtype=torch.long)
            last_idx = attn.sum(dim=1) - 1  # (B,)
            reps = hs[torch.arange(hs.size(0), device=hs.device), last_idx]  # (B,H)

            # pair positives: same prompt_count in batch, chain pairs
            pcs = meta["prompt_count"]
            # If prompt_count is tensor, move to CPU list
            if torch.is_tensor(pcs):
                pcs_list = pcs.detach().cpu().tolist()
            else:
                pcs_list = list(pcs)

            index_by_pc: Dict[int, List[int]] = {}
            for i, pid in enumerate(pcs_list):
                index_by_pc.setdefault(pid, []).append(i)

            pairs = 0
            csum = reps.new_tensor(0.0)
            for idxs in index_by_pc.values():
                if len(idxs) < 2:
                    continue
                # chain adjacent pairs to limit compute
                for j in range(len(idxs) - 1):
                    u = reps[idxs[j]]
                    v = reps[idxs[j + 1]]
                    csum = csum + (1.0 - torch.nn.functional.cosine_similarity(u, v, dim=0))
                    pairs += 1
            if pairs > 0:
                cons_loss = csum / pairs
                ce = ce + lam * cons_loss

        return ce
    
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool = False,
        ignore_keys=None,
    ):
        # Use the same filtering + custom loss path as training
        model.eval()
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)

        if prediction_loss_only:
            return (loss, None, None)

        # Optionally pass labels back if present (not required)
        labels = None
        if isinstance(inputs, dict):
            labels = inputs.get("labels", None)
            # ORPO pair stage: no single labels tensor; leave None
        return (loss, None, labels)


    # Trainer API overrides
    def compute_loss(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], return_outputs=False):
        """
        Unified compute_loss for both SFT and ORPO stages.
        Removes non-model keys before calling forward.
        """
        stage = getattr(self.args, "stage", "sft")
        # Make a shallow copy to safely pop meta-keys
        inputs = dict(inputs)

        # Separate out keys that the model can accept
        model_keys = {"input_ids", "labels", "attention_mask"}
        model_inputs = {k: v for k, v in inputs.items() if k in model_keys}
        meta_inputs = {k: v for k, v in inputs.items() if k not in model_keys}

        if stage == "orpo":
            # ORPO expects separate chosen/rejected fields; pass the whole dict
            loss = self._orpo_loss(model, inputs)
            return (loss, None) if return_outputs else loss

        # SFT path (with optional GroupDRO + Consistency)
        # Rebuild inputs to only include model-accepted keys
        loss = self._compute_sft_loss_with_regularizers(model, {**model_inputs, **meta_inputs})
        return (loss, None) if return_outputs else loss

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch: int = None) -> torch.Tensor:
        use_lap_for_batch = self.lap_kwargs.get('use_lap', False) and \
                            random.random() < self.lap_kwargs.get('lap_p_sample', 0.5)
        stage = getattr(self.args, "stage", "sft")

        # LAP is only meaningful for SFT stage
        if (not use_lap_for_batch) or (stage != "sft"):
            model.train()
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            self.accelerator.backward(loss)
            return loss.detach()

        # LAP branch
        model.train()
        inputs = self._prepare_inputs(inputs)

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
            logging.error(f"[LAP] Failed to resolve target layer: {e}. Falling back to standard step.")
            if ckpt_enabled:
                model.gradient_checkpointing_enable()
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            self.accelerator.backward(loss)
            return loss.detach()

        # Probe to get the layer *input* tensor shape
        inp_buf = {}
        def size_probe(module, p_inputs, kwargs):
            inp_buf["h"] = p_inputs[0].detach()

        h_probe = target_layer.register_forward_pre_hook(size_probe, with_kwargs=True)
        with torch.no_grad():
            out0 = self.compute_loss(model, inputs)  # baseline loss (already includes regularizers)
        h_probe.remove()

        if "h" not in inp_buf:
            logging.error("[LAP] Pre-hook size probe failed; falling back to standard step.")
            if ckpt_enabled:
                model.gradient_checkpointing_enable()
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            self.accelerator.backward(loss)
            return loss.detach()

        hidden_states_l = inp_buf["h"]
        J0 = out0.detach()

        delta = (1e-3 * torch.randn_like(hidden_states_l, dtype=torch.float32,
                                         device=hidden_states_l.device)).requires_grad_(True)
        log_lambda = torch.tensor(0.0, device=hidden_states_l.device, requires_grad=True)

        try:
            for _ in range(t_inner):
                def add_delta_hook(module, p_inputs, kwargs):
                    d = delta
                    if d.dtype != p_inputs[0].dtype or d.device != p_inputs[0].device:
                        d = d.to(dtype=p_inputs[0].dtype, device=p_inputs[0].device)
                    return (p_inputs[0] + d,), kwargs

                h = target_layer.register_forward_pre_hook(add_delta_hook, with_kwargs=True)
                J_delta = self.compute_loss(model, inputs)
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
                    J_delta_updated = self.compute_loss(model, inputs)
                    h.remove()
                    violation = (J_delta_updated - J0).abs() - epsilon
                    log_lambda.add_(lambda_lr * torch.exp(log_lambda) * violation)

                del J_delta, J_delta_updated, lagrangian, loss_constraint, violation
                torch.cuda.empty_cache()

            final_delta = delta.detach()

        except Exception as e:
            logging.error(f"[LAP] Inner loop failed; falling back to standard step. Error: {e}", exc_info=True)
            if ckpt_enabled:
                model.gradient_checkpointing_enable()
            torch.cuda.empty_cache()
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            self.accelerator.backward(loss)
            return loss.detach()

        model.train()
        try:
            def add_final_delta_hook(module, p_inputs, kwargs):
                d = final_delta
                if d.dtype != p_inputs[0].dtype or d.device != p_inputs[0].device:
                    d = d.to(dtype=p_inputs[0].dtype, device=p_inputs[0].device)
                return (p_inputs[0] + d,), kwargs

            h = target_layer.register_forward_pre_hook(add_final_delta_hook, with_kwargs=True)
            with self.compute_loss_context_manager():
                outer_loss = self.compute_loss(model, inputs)
        finally:
            if 'h' in locals():
                h.remove()
            if ckpt_enabled:
                model.gradient_checkpointing_enable()

        if self.args.n_gpu > 1:
            outer_loss = outer_loss.mean()
        self.accelerator.backward(outer_loss)
        return outer_loss.detach()


# CLI
def make_arg_parser():
    p = argparse.ArgumentParser(description="LoRA fine‑tuning on paraphrase robustness (canonical outputs)")

    # DATA & MODEL
    p.add_argument("--data_paths", nargs="+", required=True,
                   help="One or more flattened prompts JSONs (e.g., SECOND and FIRST).")
    p.add_argument("--styles_paths", nargs="*", default=[],
                   help="Optional JSON files listing instruct_* keys for each dataset in --data_paths (JSON arrays). "
                        "Length can be 0 (use all), 1 (applied to all), or equal to len(--data_paths). "
                        "instruction_original is always included.")
    p.add_argument("--model_path", default="f_finetune/model")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", default="gemma_paraphrx")

    # ANSWERS (for Stage-2 ORPO)
    p.add_argument("--answers_path", default=None,
                   help="Answers JSON (parallel to prompts) needed for --stage orpo to build chosen/rejected pairs.")
    p.add_argument("--stage", default="sft", choices=["sft", "orpo"],
                   help="Stage: 'sft' (default) for supervised canonicalization; 'orpo' for preference optimization.")

    # Legacy selector (kept for compatibility; overridden by --styles_paths if provided)
    p.add_argument("--instruct_types", nargs="+", default=[],
                   help="Fallback: space‑separated instruct_* keys to include uniformly for ALL datasets.")

    p.add_argument("--val_pct", type=float, default=0.05)
    p.add_argument("--test_pct", type=float, default=0.05)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--max_answer_tokens", type=int, default=512,
                   help="Hard cap for target answer tokens. Used during tokenisation and as a sensible inference cap.")
    p.add_argument("--weight_decay", type=float, default=0.0)  # LoRA params only

    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules",
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help='Comma-separated module names for LoRA. Use "all" or "none" for full fine-tune.')
    p.add_argument("--lora_layer_idx", type=int, default=None,
                   help="If set, enable LoRA only on this transformer layer index (e.g., 6).")

    p.add_argument("--use_deepspeed", action="store_true")
    p.add_argument("--deepspeed_config", default="ds_zero2.json")
    p.add_argument("--bnb_8bit_optim", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2,
                   help="Keep at most N checkpoints. Use ≥2 when load_best_model_at_end=True.")
    p.add_argument("--model_only_checkpoints", action="store_true",
                   help="After each checkpoint save, delete optimizer/scheduler/ZeRO files to keep checkpoints small.")
    p.add_argument("--logging_steps", type=int, default=100, help="Log training loss every N steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_checkpoint", default=None,
                   help=("Path to a Trainer checkpoint directory to resume from "
                         "(e.g. checkpoint-6600). If omitted, training starts from scratch."))
    p.add_argument("--wandb_project", default="paraphrx_ft_50k")

    p.add_argument("--early_stopping_patience", type=int, default=9,
                   help="Number of evals with no improvement before stopping")

    # Alternate path: pre-tokenized HF dataset (unchanged)
    p.add_argument("--use_pretokenized", action="store_true",
                   help="Load a preprocessed HF dataset (load_from_disk) instead of JSON + in-script tokenization.")
    p.add_argument("--tokenized_data_path", default=None,
                   help="Path for datasets.load_from_disk when --use_pretokenized is set.")

    # LAP toggles
    p.add_argument("--use_lap", action="store_true", help="Enable LAP training steps.")
    p.add_argument("--lap_layer", type=int, default=12)
    p.add_argument("--lap_t_inner", type=int, default=3)
    p.add_argument("--lap_p_sample", type=float, default=0.5)
    p.add_argument("--lap_epsilon", type=float, default=0.05)
    p.add_argument("--lap_delta_lr", type=float, default=1e-2)
    p.add_argument("--lap_lambda_lr", type=float, default=1e-3)

    # NEW: consistency & GroupDRO
    p.add_argument("--consistency_lambda", type=float, default=0.0,
                   help="Weight for paraphrase-invariance regularizer (0=off).")
    p.add_argument("--use_groupdro", action="store_true",
                   help="Enable GroupDRO weighting by style (short sweep recommended).")
    p.add_argument("--groupdro_eta", type=float, default=0.05,
                   help="Learning rate for GroupDRO log-weights.")

    # NEW: ORPO
    p.add_argument("--orpo_beta", type=float, default=0.1,
                   help="β temperature for ORPO margin.")

    # NEW: style rebalance & hard mining
    p.add_argument("--rebalance_by_style", action="store_true",
                   help="Rebalance training set so each style contributes equally (with optional upweight).")
    p.add_argument("--hard_mine_json", default=None,
                   help="Optional JSON of prompt_count IDs to upsample aggressively during rebalance.")

    # quick inspection helper
    p.add_argument("--debug_n_samples", type=int, default=0,
                   help="If >0, print N random tokenised samples and exit")
    p.add_argument("--debug_seed", type=int, default=123)

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
        "train_examples": len(trainer.train_dataset),
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
        # common deepspeed sharded state names
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


# Rebalance helper (style-level equalization + optional hard-mined upweight)
def rebalance_by_style(ds: Dataset, styles_col: str = "style",
                       hard_ids: set[int] | None = None, hard_mult: float = 3.0,
                       seed: int = 123) -> Dataset:
    from collections import defaultdict
    rng = np.random.default_rng(seed)

    # bucket indices by style and by (style,prompt_count) for hard upweight
    idxs_by_style = defaultdict(list)
    pc = ds["prompt_count"] if "prompt_count" in ds.column_names else [None] * len(ds)
    for i, s in enumerate(ds[styles_col]):
        idxs_by_style[s].append(i)

    maxn = max(len(v) for v in idxs_by_style.values())
    new_idxs = []

    for s, idxs in idxs_by_style.items():
        need = maxn
        chosen = []
        if len(idxs) >= need:
            chosen = rng.choice(idxs, size=need, replace=False).tolist()
        else:
            reps = rng.choice(idxs, size=(need - len(idxs)), replace=True).tolist()
            chosen = idxs + reps

        # upweight hard prompt_counts if provided
        if hard_ids and "prompt_count" in ds.column_names:
            extra = []
            for i in chosen:
                if ds[i]["prompt_count"] in hard_ids:
                    extra.append(i)
            # multiply count
            extra = extra * int(max(1, hard_mult - 1))
            chosen = chosen + extra

        new_idxs.extend(chosen)

    rng.shuffle(new_idxs)
    return ds.select(new_idxs)


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

        # if a directory with dataset_dict.json -> treat as load_from_disk dataset
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

    # Auto-enable pretokenized mode if an HF dataset dir was passed
    if hf_dataset_dir and not args.use_pretokenized:
        args.use_pretokenized = True
        args.tokenized_data_path = hf_dataset_dir
        logging.info("Auto-enabled --use_pretokenized with --tokenized_data_path=%s", hf_dataset_dir)

    logging.info("Datasets: %s", dataset_paths if dataset_paths else ["<none (using pretokenized)>"])

    # Styles selection per dataset
    styles_lists: List[List[str]] = []
    if args.styles_paths:
        if len(args.styles_paths) == 1 and len(dataset_paths) >= 1:
            styles = _load_styles_list(args.styles_paths[0])
            styles_lists = [styles for _ in dataset_paths]
        elif len(args.styles_paths) == len(dataset_paths):
            styles_lists = [_load_styles_list(p) for p in args.styles_paths]
        else:
            raise ValueError("--styles_paths must have length 0, 1, or match len(--data_paths).")
        logging.info("Dataset-specific styles loaded for %d datasets.", len(styles_lists))
    else:
        # fallback to --instruct_types (applied globally)
        if args.instruct_types:
            styles_lists = [args.instruct_types for _ in dataset_paths]
        else:
            styles_lists = [[] for _ in dataset_paths]

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

    # Decide LoRA vs full FT
    tm = (args.target_modules or "").strip()
    tm_lower = tm.lower()
    if tm_lower in {"all", "none"}:
        full_ft = True
        logging.info("--target_modules %s -> running full-parameter fine-tune (no LoRA).", tm_lower)
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
        logging.info("Full‑parameter fine‑tune (no LoRA)")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # needed for checkpointing
    model.config.eos_token_id = list({tokenizer.eos_token_id, EOT_ID})

    # DATA PATHS
    using_pretok = bool(args.use_pretokenized and args.tokenized_data_path)
    if not using_pretok:
        if args.stage == "orpo" and not args.answers_path:
            raise ValueError("--stage orpo requires --answers_path to construct preference pairs.")

        # SFT path (default) – always map paraphrases to canonical 'output'
        if args.stage == "sft":
            examples = load_examples_multi(dataset_paths, styles_per_path=styles_lists, use_para_ans=False)

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
                return

            raw_ds = Dataset.from_list([dataclasses.asdict(e) for e in examples])
            train_raw, val_raw, test_raw = three_way_split(
                raw_ds, val_pct=args.val_pct, test_pct=args.test_pct, seed=args.seed
            )
            logging.info("raw sizes – train: %d | val: %d | test: %d", len(train_raw), len(val_raw), len(test_raw))

            # Recommend a concise generation cap for later inference
            try:
                p95_len = _p95_answer_length(tokenizer, raw_ds)
                logging.info("Answer length p95 (tokens): %d — a good max_new_tokens cap.", p95_len)
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                with open(Path(args.output_dir) / "p95_answer_tokens.txt", "w") as fh:
                    fh.write(str(p95_len) + "\n")
            except Exception as _e:
                logging.warning("Could not compute p95 answer length: %s", _e)

            # Optional style rebalance & hard-mined upweight
            hard_ids = None
            if args.hard_mine_json and Path(args.hard_mine_json).exists():
                try:
                    hard_ids = set(json.loads(Path(args.hard_mine_json).read_text()))
                except Exception as e:
                    logging.warning("Failed to read --hard_mine_json: %s", e)

            if args.rebalance_by_style:
                train_raw = rebalance_by_style(train_raw, hard_ids=hard_ids)

            def batch_tokenise(batch):
                iids, lbls = [], []
                for pc, ins, inp, ans, sty in zip(
                    batch["prompt_count"],
                    batch["instruction"],
                    batch["inp"],
                    batch["answer"],
                    batch["style"],
                ):
                    tok = tokenise_example(Example(pc, ins, inp, ans, sty))
                    iids.append(tok["input_ids"])
                    lbls.append(tok["labels"])
                return {
                    "input_ids": iids,
                    "labels": lbls,
                    # keep meta for regularizers
                    "style": batch["style"],
                    "prompt_count": batch["prompt_count"],
                }

            train_ds = train_raw.map(batch_tokenise, batched=True, remove_columns=train_raw.column_names)
            val_ds = val_raw.map(batch_tokenise, batched=True, remove_columns=val_raw.column_names)
            test_ds = test_raw.map(batch_tokenise, batched=True, remove_columns=test_raw.column_names)
            logging.info("tokenised – train: %d | val: %d | test: %d", len(train_ds), len(val_ds), len(test_ds))

            collator = DataCollatorForSeq2Seq(
                tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
            )

        else:
            # ORPO stage: build pairs from prompts+answers (chosen = original answer, rejected = paraphrase answer)
            # Loader expects a single prompts JSON in --data_paths and a single answers JSON in --answers_path
            if len(dataset_paths) != 1:
                logging.warning("For --stage orpo, please pass exactly one prompts JSON via --data_paths. Using the first.")
            prompts_path = dataset_paths[0]
            with open(prompts_path, "r", encoding="utf-8") as fh:
                prompts_data = json.load(fh)
            with open(args.answers_path, "r", encoding="utf-8") as fh:
                answers_data = json.load(fh)
            ans_map = {a["prompt_count"]: a for a in answers_data}

            # determine styles to use for this dataset
            styles = styles_lists[0] if styles_lists else []
            if not styles:
                # default to all instruct_* keys present in the first item
                allk = [k for k in prompts_data[0].keys() if k.startswith("instruct_")]
                styles = allk

            pairs = []
            for it in prompts_data:
                pc = it["prompt_count"]
                inp = it.get("input", "")
                can_inst = it.get("instruction_original", "")
                if pc not in ans_map:
                    continue
                ansrec = ans_map[pc]
                chosen = ansrec.get("instruction_original", "")
                if not chosen:
                    continue
                for k in styles:
                    pi = it.get(k, "")
                    rej = ansrec.get(k, "")
                    if not pi or not rej:
                        continue
                    pairs.append({
                        "prompt_count": pc,
                        "instruction_chosen": can_inst,  # not used, but kept for debug
                        "instruction_rejected": pi,
                        "inp": inp,
                        "chosen": chosen,
                        "rejected": rej,
                        "style": k,
                        "weight": 1.0,  # optional external weights can be added later
                    })

            # Tokenize pairs into model inputs (chosen and rejected)
            def tok_pair(batch):
                iids_c, labs_c, am_c = [], [], []
                iids_r, labs_r, am_r = [], [], []
                styles, pcs, wts = [], [], []
                for pc, ins_c, ins_r, inp, ch, rj, sty, wt in zip(
                    batch["prompt_count"],
                    batch["instruction_chosen"],
                    batch["instruction_rejected"],
                    batch["inp"],
                    batch["chosen"],
                    batch["rejected"],
                    batch["style"],
                    batch["weight"]
                ):
                    # chosen
                    ptxt_c = build_chat_prompt(ins_c, inp)
                    pid_c = tokenizer(ptxt_c, add_special_tokens=False)
                    aid_c = tokenizer(ch, add_special_tokens=False, truncation=True, max_length=MAX_ANS_TOKENS)
                    input_ids_c = pid_c["input_ids"] + aid_c["input_ids"] + [EOT_ID]
                    labels_c = [-100] * len(pid_c["input_ids"]) + aid_c["input_ids"] + [EOT_ID]
                    # rejected
                    ptxt_r = build_chat_prompt(ins_r, inp)
                    pid_r = tokenizer(ptxt_r, add_special_tokens=False)
                    aid_r = tokenizer(rj, add_special_tokens=False, truncation=True, max_length=MAX_ANS_TOKENS)
                    input_ids_r = pid_r["input_ids"] + aid_r["input_ids"] + [EOT_ID]
                    labels_r = [-100] * len(pid_r["input_ids"]) + aid_r["input_ids"] + [EOT_ID]

                    iids_c.append(input_ids_c); labs_c.append(labels_c)
                    iids_r.append(input_ids_r); labs_r.append(labels_r)
                    styles.append(sty); pcs.append(pc); wts.append(float(wt))

                return {
                    "input_ids_chosen": iids_c, "labels_chosen": labs_c,
                    "input_ids_rejected": iids_r, "labels_rejected": labs_r,
                    "style": styles, "prompt_count": pcs, "weight": wts,
                }

            raw_pairs = Dataset.from_list(pairs)
            train_raw, val_raw, test_raw = three_way_split(raw_pairs, val_pct=args.val_pct, test_pct=args.test_pct, seed=args.seed)

            train_ds = train_raw.map(tok_pair, batched=True, remove_columns=raw_pairs.column_names)
            val_ds = val_raw.map(tok_pair, batched=True, remove_columns=raw_pairs.column_names)
            test_ds = test_raw.map(tok_pair, batched=True, remove_columns=raw_pairs.column_names)

            # Simple pad collator for pair tensors
            def collate_pairs(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
                # flatten lists and pad individually for chosen/rejected
                def pad_stack(key):
                    seqs = [torch.tensor(f[key], dtype=torch.long) for f in features]
                    return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad_token_id)

                batch = {
                    "input_ids_chosen": pad_stack("input_ids_chosen"),
                    "labels_chosen": pad_stack("labels_chosen"),
                    "input_ids_rejected": pad_stack("input_ids_rejected"),
                    "labels_rejected": pad_stack("labels_rejected"),
                }
                # attention masks
                batch["attention_mask_chosen"] = (batch["input_ids_chosen"] != tokenizer.pad_token_id).long()
                batch["attention_mask_rejected"] = (batch["input_ids_rejected"] != tokenizer.pad_token_id).long()
                # meta
                batch["style"] = [f["style"] for f in features]
                batch["prompt_count"] = torch.tensor([f["prompt_count"] for f in features], dtype=torch.long)
                batch["weight"] = torch.tensor([f.get("weight", 1.0) for f in features], dtype=torch.float32)
                return batch

            collator = collate_pairs

    else:
        # Alternate path: use pre-tokenized dataset
        logging.info(f"Loading pre-tokenized data from {args.tokenized_data_path}")
        tokenized = load_from_disk(args.tokenized_data_path)
        train_ds = tokenized["train"]
        val_ds = tokenized.get("validation", None) or tokenized.get("val", None)
        test_ds = tokenized.get("test", None)
        if val_ds is None:
            logging.warning("No 'validation' split found; using a small slice of train for eval.")
            val_ds = train_ds.select(range(min(len(train_ds), 1024)))
        logging.info("tokenised – train: %d | val: %d | test: %d",
                     len(train_ds), len(val_ds), (len(test_ds) if test_ds else 0))
        collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
        )

    # DeepSpeed option
    ds_cfg = args.deepspeed_config if args.use_deepspeed else None
    if args.use_deepspeed:
        try:
            importlib.import_module("deepspeed")
        except ImportError:
            logging.warning("DeepSpeed not available, disabling")
            ds_cfg = None

    # Inject stage into TrainingArguments so the Trainer can branch
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
        remove_unused_columns=False,
    )

    # attach custom attributes used in Trainer
    setattr(targs, "stage", args.stage)
    setattr(targs, "consistency_lambda", args.consistency_lambda)
    setattr(targs, "use_groupdro", args.use_groupdro)
    setattr(targs, "groupdro_eta", args.groupdro_eta)
    setattr(targs, "orpo_beta", args.orpo_beta)

    class StepDigest(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return

            # print train loss every logging_steps
            if "loss" in logs and state.global_step % args.logging_steps == 0:
                logging.info(
                    "step %4d | train_loss %.4f | lr %.3g",
                    state.global_step,
                    logs["loss"],
                    logs.get("learning_rate", float("nan")),
                )

            # print validation loss whenever it appears
            if "eval_loss" in logs:
                logging.info(
                    "step %4d | eval_loss  %.4f | perplexity %.2f",
                    state.global_step,
                    logs["eval_loss"],
                    math.exp(logs["eval_loss"]),
                )

    class ModelOnlyCheckpointCallback(TrainerCallback):
        """Delete heavy optimizer/scheduler (and DS ZeRO) files right after each save"""
        def on_save(self, args, state, control, **kwargs):
            try:
                # Find the most recent checkpoint and prune it
                ckpt_dir = find_latest_checkpoint_dir(Path(args.output_dir))
                if ckpt_dir is None:
                    return
                prune_checkpoint_dir(ckpt_dir)
            except Exception as e:
                logging.warning("Prune failed: %s", e)

    if full_ft:
        for p in model.parameters():
            p.requires_grad_(True)

    # Trainer
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
    lap_cfg = {k: getattr(args, k) for k in [
        "use_lap", "lap_layer", "lap_t_inner", "lap_p_sample", "lap_epsilon", "lap_delta_lr", "lap_lambda_lr"
    ]}
    trainer_kwargs["lap_kwargs"] = lap_cfg

    trainer = ParaphrxTrainer(**trainer_kwargs)

    logging.info("Step-0 evaluation (pre-fine-tuning)")
    init_metrics = trainer.evaluate()
    logging.info("step 0 | eval_loss %.4f | ppl %.2f", init_metrics["eval_loss"], math.exp(init_metrics["eval_loss"]))
    if wandb.run:
        wandb.log({"eval_loss/step0": init_metrics["eval_loss"]})

    # Train
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
    # Save final
    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logging.info("Training done: model & tokenizer in %s", final_dir)

    # style-wise validation loss (SFT only; ORPO val metric is margin)
    if args.stage == "sft":
        from collections import defaultdict

        # Build a loader that keeps meta columns out of model forward
        val_loader = torch.utils.data.DataLoader(
            val_ds.remove_columns([]),  # keep all; Trainer will pop in compute_loss
            batch_size=args.batch_size * 2,
            shuffle=False,
            collate_fn=trainer.data_collator,
        )
        styles = val_ds["style"] if "style" in val_ds.column_names else ["_all"] * len(val_ds)  # parallel list

        model.eval()
        losses_by_style = defaultdict(list)
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                # compute per-batch scalar loss through trainer path
                loss = trainer.compute_loss(model, batch)
                bsz = batch["input_ids"].size(0) if "input_ids" in batch else len(batch["style"])
                sl = styles[idx * bsz: (idx + 1) * bsz]
                for s in sl:
                    # store scalar loss proxy for each sample in the batch (approximate)
                    losses_by_style[s].append(float(loss.detach().cpu().item()))

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
