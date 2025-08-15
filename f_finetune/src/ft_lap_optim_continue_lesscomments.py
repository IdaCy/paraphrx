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
import json
import logging
import math
import os
import random
import sys
import gc
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import wandb
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments,
    EarlyStoppingCallback, logging as hf_logging,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Setup
os.environ.setdefault("TQDM_MININTERVAL", "30")
hf_logging.set_verbosity_warning()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)

# Custom Trainer for LAP

class LAPTtrainer(Trainer):
    """
    A custom Hugging Face Trainer that implements Latent Adversarial Paraphrasing (LAP)
    """
    def __init__(self, *args, **kwargs):
        self.lap_kwargs = kwargs.pop('lap_kwargs', {})
        super().__init__(*args, **kwargs)
        if self.lap_kwargs.get('use_lap', False):
            logging.info("LAP Trainer initialized with custom training step.")
            logging.info(f"LAP Config: {json.dumps(self.lap_kwargs, indent=2)}")

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Overrides the default training step
        With a probability of 'lap_p_sample', performs a LAP step
        Otherwise, performs a standard supervised fine-tuning (SFT) step
        """
        use_lap_for_batch = self.lap_kwargs.get('use_lap', False) and \
                            random.random() < self.lap_kwargs.get('lap_p_sample', 0.5)

        model.train() # Default mode
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            if use_lap_for_batch:
                loss = self.lap_training_step(model, inputs)
            else:
                # Standard SFT forward pass
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)
        return loss.detach()

    def lap_training_step(self, model, inputs):
        """
        Performs one step of Latent Adversarial Paraphrasing
        This involves an inner loop to find an adversarial perturbation 'delta'
        and an outer loop (the main model update) to train against it
        """
        # LAP Hyperparameters
        t_inner = self.lap_kwargs['lap_t_inner']
        epsilon = self.lap_kwargs['lap_epsilon']
        delta_lr = self.lap_kwargs['lap_delta_lr']
        lambda_lr = self.lap_kwargs['lap_lambda_lr']
        layer_idx = self.lap_kwargs['lap_layer']

        # model is fixed during the inner loop to find the perturbation
        model.eval()

        # Inner Loop: Find Adversarial Perturbation 'delta'
        hook_handle = None
        try:
            # 1. get original loss (J0) and hidden states without any perturbation
            with torch.no_grad():
                # To get hidden states, we must pass them as inputs to the base model
                outputs_orig = model(
                    **inputs,
                    output_hidden_states=True
                )
                J0 = outputs_orig.loss.detach()
                # Hidden states are a tuple, one for each layer
                hidden_states_l = outputs_orig.hidden_states[layer_idx].detach()

            # 2. initialise perturbation 'delta' and Lagrange multiplier 'lambda'
            delta = torch.zeros_like(hidden_states_l, requires_grad=True)
            # We optimize log_lambda for numerical stability
            log_lambda = torch.tensor(0.0, device=model.device, requires_grad=True)

            # 3. inner optimization loop (min-max game)
            for i in range(t_inner):
                # need to re-attach the hook in each loop iteration
                # as it's removed in the finally block
                target_layer = model.get_base_model().model.layers[layer_idx]
                def add_delta_hook(module, p_inputs, kwargs):
                    # Add perturbation to the input of the target layer
                    return (p_inputs[0] + delta,), kwargs
                hook_handle = target_layer.register_forward_pre_hook(add_delta_hook, with_kwargs=True)

                # Update delta (minimise Lagrangian)
                # Calculate loss with perturbation
                J_delta = self.compute_loss(model, inputs)
                hook_handle.remove()

                # Lagrangian as per paper (Eq. 6)
                # L(δ, λ) = -||δ||₂ + λ * (|J_δ - J₀| - ε)
                # want to minimise L wrt δ, and maximise wrt λ
                norm_delta = torch.norm(delta, p=2)
                loss_constraint = torch.abs(J_delta - J0) - epsilon
                lagrangian = -norm_delta + torch.exp(log_lambda).detach() * loss_constraint

                # Minimise L wrt δ -> Gradient Descent on L
                # We need the gradient of L wrt δ
                delta.grad = None
                lagrangian.backward()
                delta.data.add_(delta.grad, alpha=-delta_lr) # delta = delta - lr * grad
                delta.grad = None # Zero out grad for next iteration

                # Update lambda (maximise Lagrangian)
                # done via gradient ascent on L wrt λ
                # The gradient of L wrt log_lambda is λ * (|J_δ - J₀| - ε)
                log_lambda.grad = None
                with torch.no_grad():
                    # Re-evaluate constraint with updated delta
                    hook_handle = target_layer.register_forward_pre_hook(add_delta_hook, with_kwargs=True)
                    J_delta_updated = self.compute_loss(model, inputs)
                    hook_handle.remove()
                    loss_constraint_updated = torch.abs(J_delta_updated - J0) - epsilon

                log_lambda_grad = torch.exp(log_lambda) * loss_constraint_updated
                # Maximise L wrt λ -> Gradient Ascent
                log_lambda.data.add_(log_lambda_grad, alpha=lambda_lr)
                
                # Logging for debugging
                if self.state.global_step % self.args.logging_steps == 0:
                     logging.info(
                        f"[LAP Inner {i+1}/{t_inner}] "
                        f"J0: {J0:.4f}, J_delta: {J_delta_updated:.4f}, "
                        f"delta_norm: {norm_delta.item():.4f}, "
                        f"lambda: {torch.exp(log_lambda).item():.4f}"
                    )
                # Cleanup
                del J_delta, J_delta_updated, lagrangian, norm_delta, log_lambda_grad
                torch.cuda.empty_cache()


            final_delta = delta.detach()

        except Exception as e:
            logging.error(f"LAP inner loop failed, falling back to SFT for this step. Error: {e}", exc_info=True)
            if hook_handle: hook_handle.remove()
            torch.cuda.empty_cache()
            model.train()
            return self.compute_loss(model, inputs)  # Fallback to standard training

        # Outer Loop: Update Model Parameters
        # Now, train the model to be robust to this *fixed* perturbation
        model.train()
        hook_handle = None
        try:
            target_layer = model.get_base_model().model.layers[layer_idx]
            def add_final_delta_hook(module, p_inputs, kwargs):
                return (p_inputs[0] + final_delta,), kwargs
            hook_handle = target_layer.register_forward_pre_hook(add_final_delta_hook, with_kwargs=True)
            
            # This is the final loss that will be used for the model's gradient update
            outer_loss = self.compute_loss(model, inputs)
        finally:
            # CRITICAL: Always remove the hook
            if hook_handle: hook_handle.remove()

        return outer_loss


def make_arg_parser():
    p = argparse.ArgumentParser(description="Fine-tuning with Latent Adversarial Paraphrasing (LAP)")
    # Paths
    p.add_argument("--tokenized_data_path", required=True, help="Path to the pre-tokenized dataset directory created by the tokenization script.")
    p.add_argument("--model_path", required=True, help="Path to the base model (e.g., gemma-2-2b).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a specific checkpoint to resume training from.")

    # hyperparameters
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--weight_decay", type=float, default=0.01)

    # Model config (LoRA or Full-FT)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj", help="Comma-separated list of modules for LoRA. Use 'none' for full-parameter fine-tuning.")

    # Hardware and precision
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 precision.")
    p.add_argument("--bnb_8bit_optim", action="store_true", help="Use 8-bit paged AdamW optimizer (for LoRA).")

    # LAP-Specific args
    p.add_argument("--use_lap", action="store_true", help="Enable Latent Adversarial Paraphrasing.")
    p.add_argument("--lap_layer", type=int, default=12, help="Index of the transformer layer to apply perturbation.")
    p.add_argument("--lap_t_inner", type=int, default=3, help="Number of inner loop optimization steps for LAP.")
    p.add_argument("--lap_p_sample", type=float, default=0.5, help="Probability of applying LAP to a batch.")
    p.add_argument("--lap_epsilon", type=float, default=0.05, help="Constraint on the LM loss increase.")
    p.add_argument("--lap_delta_lr", type=float, default=1e-2, help="Learning rate for the perturbation delta.")
    p.add_argument("--lap_lambda_lr", type=float, default=1e-3, help="Learning rate for the Lagrange multiplier lambda.")

    # Logging and saving
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
    # Reconfigure logging to write to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(file_handler)

    run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    logging.info(f"Starting run '{args.run_name}' with config:\n{json.dumps(vars(args), indent=2)}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    logging.info(f"Loading pre-tokenized data from {args.tokenized_data_path}")
    try:
        tokenized_datasets = load_from_disk(args.tokenized_data_path)
    except FileNotFoundError:
        logging.error(f"Tokenized data not found at {args.tokenized_data_path}. Please run the tokenization script first.")
        sys.exit(1)

    train_ds, val_ds = tokenized_datasets["train"], tokenized_datasets["test"]
    logging.info(f"Loaded {len(train_ds)} training and {len(val_ds)} validation examples.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Model Loading (Full-FT vs LoRA)
    is_full_ft = not args.target_modules or args.target_modules.lower() in {"all", "none"}
    
    # Use the checkpoint path for the model if resuming, otherwise use base model path
    model_load_path = args.resume_from_checkpoint if args.resume_from_checkpoint else args.model_path

    if is_full_ft:
        logging.info(f"Configuring for FULL-PARAMETER fine-tuning from '{model_load_path}'.")
        model = AutoModelForCausalLM.from_pretrained(
            model_load_path,
            device_map="auto",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        )
    else:
        logging.info(f"Configuring for LoRA fine-tuning from '{model_load_path}'.")
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

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    if not is_full_ft:
        mods = [m.strip() for m in args.target_modules.split(",")]
        lcfg = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            target_modules=mods, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lcfg)
        logging.info("Applied LoRA to modules:")
        model.print_trainable_parameters()

    # Training
    targs = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        num_train_epochs=args.num_epochs,
        optim="paged_adamw_8bit" if not is_full_ft and args.bnb_8bit_optim else "adamw_torch",
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        lap_kwargs={k: v for k, v in vars(args).items() if k.startswith('lap_')}
    )
    
    # pass resume_from_checkpoint argument to the train method
    if args.resume_from_checkpoint:
        logging.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
    else:
        logging.info("Starting training from scratch...")
        
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logging.info("Training finished.")

    final_path = output_dir / 'final'
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logging.info(f"Final model saved to {final_path}")
    
    if run:
        run.finish()

if __name__ == '__main__':
    main()
