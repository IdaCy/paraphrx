#!/usr/bin/env python
"""
supports both full-parameter and LoRA fine-tuning 
+ can apply the LAP technique to improve paraphr robustness
- based on Fu & Barez (2025)

latest version of FT LAP SFT!
'continue' for allowing to go on with stopped checkpoints

Full-parameter fine-tune with LAP:
srun python $RUN_SCRIPT \
  --tokenized_data_path ./data/tokenized \
  --model_path ./models/gemma-2-2b \
  --output_dir ./runs/gemma2b_full_lap \
  --run_name gemma2b_full_lap \
  --target_modules none \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 8e-6 \
  --bf16 \
  --use_lap --lap_layer 12 --lap_t_inner 3

LoRA fine-tune with LAP:
srun python $RUN_SCRIPT \
  --tokenized_data_path ./data/tokenized \
  --model_path ./models/gemma-2-2b \
  --output_dir ./runs/gemma2b_lora_lap \
  --run_name gemma2b_lora_lap \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --lora_rank 16 --lora_alpha 32 \
  --bnb_8bit_optim --bf16 \
  --use_lap --lap_layer 12
"""
import argparse
import contextlib
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any, Iterable

import numpy as np
import torch
import wandb
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments,
    EarlyStoppingCallback, logging as hf_logging,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import matplotlib
matplotlib.use('Agg')  # headless-safe
import matplotlib.pyplot as plt


# Logging / setup
os.environ.setdefault("TQDM_MININTERVAL", "30")
hf_logging.set_verbosity_warning()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
    force=True
)


# Helpers: logging, plots, layer resolution
class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_world_process_zero:
            if 'loss' in logs and 'learning_rate' in logs:
                logging.info(
                    f"step {state.global_step:5d} | train_loss {logs['loss']:.4f} | "
                    f"lr {logs['learning_rate']:.2e}"
                )
            if 'eval_loss' in logs:
                try:
                    ppl = math.exp(logs['eval_loss'])
                except OverflowError:
                    ppl = float('inf')
                logging.info(
                    f"step {state.global_step:5d} | eval_loss  {logs['eval_loss']:.4f} | "
                    f"perplexity {ppl:.2f}"
                )


def write_summary_file(output_dir: Path, trainer: Trainer, train_ds, val_ds):
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in trainer.model.parameters())

    eval_logs = [log for log in trainer.state.log_history if 'eval_loss' in log]
    if eval_logs:
        best_eval = min(eval_logs, key=lambda x: x['eval_loss'])
        final_val_loss = best_eval['eval_loss']
        final_val_ppl = math.exp(final_val_loss)
    else:
        final_val_loss = float('nan')
        final_val_ppl = float('nan')

    summary = (
        f"train_examples: {len(train_ds)}\n"
        f"val_examples: {len(val_ds)}\n"
        f"trainable_params: {trainable_params}\n"
        f"total_params: {total_params}\n"
        f"final_val_loss: {final_val_loss}\n"
        f"final_val_ppl: {final_val_ppl}\n"
    )
    (output_dir / "summary.txt").write_text(summary)
    logging.info(f"Summary file saved to {output_dir/'summary.txt'}")


def generate_loss_curve(output_dir: Path, trainer: Trainer):
    hist = trainer.state.log_history
    tr_steps = [h['step'] for h in hist if 'loss' in h]
    tr_loss  = [h['loss'] for h in hist if 'loss' in h]
    ev_steps = [h['step'] for h in hist if 'eval_loss' in h]
    ev_loss  = [h['eval_loss'] for h in hist if 'eval_loss' in h]

    if not tr_loss or not ev_loss:
        logging.warning("Not enough data to plot loss curve.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(tr_steps, tr_loss, label='Train Loss', alpha=0.7)
    plt.plot(ev_steps, ev_loss, label='Validation Loss', linestyle='--', marker='o')
    plt.title(f'Training and Validation Loss — {output_dir.name}')
    plt.xlabel('Step'); plt.ylabel('Loss'); plt.grid(True); plt.legend()
    out = output_dir / "loss_curve.png"
    plt.savefig(out)
    logging.info(f"Loss curve saved to {out}")


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
        "model.layers",                # LLaMA/Mistral/Gemma style
        "model.model.layers",          # some wrapped variants
        "model.decoder.layers",        # T5/EncDec dec
        "transformer.h",               # GPT-2 style
        "gpt_neox.layers",             # GPT-NeoX
        "backbone.layers",             # some custom backbones
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


# The LAP-enabled Trainer
class LAPTtrainer(Trainer):
    """
    Trainer with Latent Adversarial Paraphrasing (LAP) training_step.
    """
    def __init__(self, *args, **kwargs):
        self.lap_kwargs = kwargs.pop('lap_kwargs', {})
        super().__init__(*args, **kwargs)
        if self.lap_kwargs.get('use_lap', False):
            logging.info("LAP Trainer initialized with custom training step.")
            logging.info("LAP Config:\n" + json.dumps(self.lap_kwargs, indent=2))

    # training step
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
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

    # LAP core
    def lap_training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        One LAP step:
          * capture layer input shape
          * optimize δ in inner loop (Lagrangian relaxation)
          * apply δ and do outer supervised step
        """
        # Hyperparams
        t_inner     = int(self.lap_kwargs['lap_t_inner'])
        epsilon     = float(self.lap_kwargs['lap_epsilon'])
        delta_lr    = float(self.lap_kwargs['lap_delta_lr'])
        lambda_lr   = float(self.lap_kwargs['lap_lambda_lr'])
        layer_idx   = int(self.lap_kwargs['lap_layer'])

        # Temporarily disable gradient checkpointing (forward hooks + inner loop can interact poorly)
        ckpt_enabled = getattr(model, "is_gradient_checkpointing", False)
        if ckpt_enabled:
            model.gradient_checkpointing_disable()

        # Resolve target layer
        try:
            base = model.get_base_model() if hasattr(model, "get_base_model") else model
            layers = resolve_transformer_layers(model)
            if not (0 <= layer_idx < len(layers)):
                raise IndexError(f"lap_layer={layer_idx} out of range [0, {len(layers)-1}]")
            target_layer = layers[layer_idx]
        except Exception as e:
            logging.error(f"[LAP] Failed to resolve target layer: {e}. Falling back to SFT.")
            if ckpt_enabled:
                model.gradient_checkpointing_enable()
            return self.compute_loss(model, inputs)

        # Probe to get the layer *input* tensor shape (more robust than output)
        inp_buf = {}
        def size_probe(module, p_inputs, kwargs):
            # p_inputs[0]: layer input hidden states [bsz, seq, dim]
            inp_buf["h"] = p_inputs[0].detach()

        h_probe = target_layer.register_forward_pre_hook(size_probe, with_kwargs=True)
        with torch.no_grad():
            out0 = model(**inputs)
        h_probe.remove()

        if "h" not in inp_buf:
            logging.error("[LAP] Pre-hook size probe failed; falling back to SFT.")
            if ckpt_enabled:
                model.gradient_checkpointing_enable()
            return self.compute_loss(model, inputs)

        hidden_states_l = inp_buf["h"]
        J0 = out0.loss.detach()

        # Initialize δ slightly away from zero to avoid zero-grad at start.
        # Keep δ in float32 for numerical stability; cast in the pre-hook on use.
        delta = (1e-3 * torch.randn_like(hidden_states_l, dtype=torch.float32,
                                         device=hidden_states_l.device)).requires_grad_(True)
        log_lambda = torch.tensor(0.0, device=hidden_states_l.device, requires_grad=True)

        # inner loop on δ
        try:
            for i in range(t_inner):

                # Pre-hook to add δ to layer input (cast to match the module's input dtype/device)
                def add_delta_hook(module, p_inputs, kwargs):
                    d = delta
                    # match dtype/device of incoming activations
                    if d.dtype != p_inputs[0].dtype or d.device != p_inputs[0].device:
                        d = d.to(dtype=p_inputs[0].dtype, device=p_inputs[0].device)
                    return (p_inputs[0] + d,), kwargs

                h = target_layer.register_forward_pre_hook(add_delta_hook, with_kwargs=True)
                J_delta = self.compute_loss(model, inputs)
                h.remove()

                # Smooth constraint to avoid zero sub-gradient at equality
                # Option A: two-sided squared (faithful to |Jδ-J0|<=ε)
                loss_constraint = (J_delta - J0).pow(2) - (epsilon ** 2)
                # Option B (alternative): hinge-squared upper bound
                # loss_constraint = torch.relu(J_delta - (J0 + epsilon)).pow(2)

                # Lagrangian (λ is treated as constant while optimizing δ)
                lagrangian = -delta.norm(p=2) + torch.exp(log_lambda).detach() * loss_constraint

                # Only update δ — do not accumulate grads on model params
                model.zero_grad(set_to_none=True)
                delta.grad = None
                lagrangian.backward()
                with torch.no_grad():
                    delta.add_(-delta_lr, delta.grad)
                delta.grad = None

                # Dual ascent on λ (no graph)
                with torch.no_grad():
                    h = target_layer.register_forward_pre_hook(add_delta_hook, with_kwargs=True)
                    J_delta_updated = self.compute_loss(model, inputs)
                    h.remove()
                    # violation for squared version:
                    violation = (J_delta_updated - J0).abs() - epsilon
                    log_lambda.add_(lambda_lr * torch.exp(log_lambda) * violation)

                # Optional debug
                if self.state.is_world_process_zero and \
                   self.state.global_step % max(1, self.args.logging_steps) == 0:
                    dn = float(delta.norm(p=2).item())
                    logging.info(
                        f"[LAP Inner {i+1}/{t_inner}] "
                        f"J0: {float(J0):.4f}, J_delta: {float(J_delta):.4f}, "
                        f"delta_norm: {dn:.4f}, lambda: {float(torch.exp(log_lambda).item()):.4f}"
                    )

                del J_delta, J_delta_updated, lagrangian, loss_constraint, violation
                torch.cuda.empty_cache()

            final_delta = delta.detach()

        except Exception as e:
            logging.error(f"[LAP] Inner loop failed; falling back to SFT. Error: {e}", exc_info=True)
            if ckpt_enabled:
                model.gradient_checkpointing_enable()
            torch.cuda.empty_cache()
            return self.compute_loss(model, inputs)

        # outer supervised step
        model.train()
        try:
            def add_final_delta_hook(module, p_inputs, kwargs):
                d = final_delta
                if d.dtype != p_inputs[0].dtype or d.device != p_inputs[0].device:
                    d = d.to(dtype=p_inputs[0].dtype, device=p_inputs[0].device)
                return (p_inputs[0] + d,), kwargs

            h = target_layer.register_forward_pre_hook(add_final_delta_hook, with_kwargs=True)
            outer_loss = self.compute_loss(model, inputs)
        finally:
            if 'h' in locals():
                h.remove()
            if ckpt_enabled:
                model.gradient_checkpointing_enable()

        return outer_loss


# CLI / main
def make_arg_parser():
    p = argparse.ArgumentParser(description="Fine-tuning with Latent Adversarial Paraphrasing (LAP)")
    # Paths
    p.add_argument("--tokenized_data_path", required=True, help="Path to dataset saved by the tokenization script.")
    p.add_argument("--model_path", required=True, help="Base or checkpoint model path.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    # SFT hyperparams
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--weight_decay", type=float, default=0.01)

    # Model config (LoRA vs full-FT)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj",
                   help="Comma-separated LoRA target modules; use 'none' or 'all' for full-FT.")

    # Hardware / precision
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--bnb_8bit_optim", action="store_true")

    # LAP parameters
    p.add_argument("--use_lap", action="store_true")
    p.add_argument("--lap_layer", type=int, default=12)
    p.add_argument("--lap_t_inner", type=int, default=3)
    p.add_argument("--lap_p_sample", type=float, default=0.5)
    p.add_argument("--lap_epsilon", type=float, default=0.05)
    p.add_argument("--lap_delta_lr", type=float, default=1e-2)
    p.add_argument("--lap_lambda_lr", type=float, default=1e-3)

    # Logging / saving
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default="paraphrx_ft_lap")
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=200)
    return p


def main():
    args = make_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"{args.run_name}.log"

    # Route logs to both file and stdout
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(logging.StreamHandler(sys.stdout))
    root.addHandler(file_handler)
    root.setLevel(logging.INFO)

    run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    logging.info(f"Starting run '{args.run_name}'")
    logging.info("Command-line args:\n" + json.dumps(vars(args), indent=2))

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    logging.info(f"Loading pre-tokenized data from {args.tokenized_data_path}")
    try:
        tokenized = load_from_disk(args.tokenized_data_path)
    except Exception as e:
        logging.error(f"Failed to load tokenized data: {e}")
        sys.exit(1)

    train_ds = tokenized["train"]
    val_ds   = tokenized.get("validation", None) or tokenized.get("val", None)
    test_ds  = tokenized.get("test", None)
    if val_ds is None:
        logging.warning("No 'validation' split found; using a small slice of train for eval.")
        val_ds = train_ds.select(range(min(len(train_ds), 1024)))

    logging.info(f"tokenised – train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds) if test_ds else 0}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    full_ft = not args.target_modules or args.target_modules.lower() in {"all", "none", "null", "false"}
    model_load_path = args.resume_from_checkpoint if args.resume_from_checkpoint else args.model_path

    if full_ft:
        logging.info(f"Configuring FULL-parameter fine-tuning from '{model_load_path}'.")
        model = AutoModelForCausalLM.from_pretrained(
            model_load_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        )
    else:
        logging.info(f"Configuring LoRA fine-tuning from '{model_load_path}'.")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_load_path, device_map="auto", quantization_config=bnb_cfg
        )
        model = prepare_model_for_kbit_training(model)

    # Enable checkpointing; non-reentrant is friendlier to hooks
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "config"):
        model.config.use_cache = False

    if not full_ft:
        mods = [m.strip() for m in args.target_modules.split(",")]
        lcfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=mods,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lcfg)
        logging.info(f"LoRA on modules: {mods}")
        # Optional: print trainable param count quietly
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            model.print_trainable_parameters()

    # Training args
    targs = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        num_train_epochs=args.num_epochs,
        optim=("paged_adamw_8bit" if (not full_ft and args.bnb_8bit_optim) else "adamw_torch"),
        bf16=args.bf16,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        report_to=['wandb'] if run else [],
        seed=args.seed
    )

    trainer = LAPTtrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            CustomLoggingCallback()
        ],
        lap_kwargs={k: v for k, v in vars(args).items() if k.startswith('lap_') or k == 'use_lap'}
    )

    if args.resume_from_checkpoint:
        logging.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
    else:
        logging.info("Starting training from scratch...")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logging.info("Training finished.")

    # Outputs
    write_summary_file(output_dir, trainer, train_ds, val_ds)
    generate_loss_curve(output_dir, trainer)

    # Per-style validation loss if available
    if 'style' in val_ds.column_names:
        styles = sorted(set(val_ds['style']))
        for sty in styles:
            sub = val_ds.filter(lambda x: x['style'] == sty)
            if len(sub) > 0:
                m = trainer.evaluate(eval_dataset=sub)
                logging.info(f"VAL-loss {sty:<25} {m['eval_loss']:.4f}")
    else:
        logging.warning("Column 'style' not found in validation set; skipping per-type eval.")

    # Final test evaluation (optional)
    if test_ds is not None:
        logging.info("Evaluating on the final test set...")
        tm = trainer.evaluate(eval_dataset=test_ds)
        logging.info(f"TEST loss {tm['eval_loss']:.4f} | ppl {math.exp(tm['eval_loss']):.2f}")

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logging.info(f"Final model + tokenizer saved to {final_dir}")

    if run:
        run.finish()


if __name__ == '__main__':
    main()
