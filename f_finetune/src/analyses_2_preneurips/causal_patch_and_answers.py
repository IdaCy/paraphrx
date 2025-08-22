"""
SCRIPT C
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# Split helper (MUST match training)
# =========================
def three_way_split(
    ds: Dataset, *, val_pct: float, test_pct: float, seed: int
) -> Tuple[set[int], set[int], set[int]]:
    """Group-wise split guaranteeing each prompt_count stays in one split."""
    import numpy as np

    rng = np.random.default_rng(seed)
    pcs = list({int(ex["prompt_count"]) for ex in ds})
    rng.shuffle(pcs)
    n = len(pcs)
    n_val = int(n * val_pct)
    n_test = int(n * test_pct)
    val_ids = set(pcs[:n_val])
    test_ids = set(pcs[n_val : n_val + n_test])
    train_ids = set(pcs[n_val + n_test :])
    return train_ids, val_ids, test_ids

# =========================
# Loading prompts + RESUME merge
# (kept exactly; used per-variant below)
# =========================
def load_examples_with_resume(
    data_path: str,
    instruct_types: List[str] | None,
    val_pct: float,
    test_pct: float,
    seed: int,
    split: str,
    max_samples: int,
    from_prompt_id: int,
    upto_prompt_id: int,
    resume_json_path: str | None,
) -> Tuple[List[Tuple[int, str, str, str]], Dict[int, Dict]]:
    """
    Returns:
        flat_queue - list of (prompt_count, key_name, prompt_text, raw_input)
                     containing ONLY the missing/empty keys to generate.
        results_map - dict keyed by prompt_count with merged existing results:
                      {"prompt_count", "input", <answer fields...>}
    """
    # Load prompts JSON
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Build Dataset for split, exactly as in training
    ds = Dataset.from_list(raw_data)
    train_ids, val_ids, test_ids = three_way_split(
        ds, val_pct=val_pct, test_pct=test_pct, seed=seed
    )
    if split == "val":
        keep_ids = val_ids
    elif split == "test":
        keep_ids = test_ids
    else:  # held-out = val ∪ test
        keep_ids = val_ids | test_ids

    # Filter by prompt_count ID ranges
    sorted_keep_ids = sorted(list(keep_ids))

    if from_prompt_id > 0:
        sorted_keep_ids = [pc for pc in sorted_keep_ids if pc >= from_prompt_id]
        logging.info("Filtering from prompt_count >= %d, %d IDs remain.", from_prompt_id, len(sorted_keep_ids))

    if upto_prompt_id > 0:
        sorted_keep_ids = [pc for pc in sorted_keep_ids if pc <= upto_prompt_id]
        logging.info("Filtering up to prompt_count <= %d, %d IDs remain.", upto_prompt_id, len(sorted_keep_ids))

    if max_samples > 0:
        sorted_keep_ids = sorted_keep_ids[:max_samples]
        logging.info("Applying max_samples=%d, final group count is %d.", max_samples, len(sorted_keep_ids))

    keep_ids = set(sorted_keep_ids)

    # Init results_map with prompt_count + input for all kept IDs
    results_map: Dict[int, Dict] = {}
    for item in raw_data:
        pc = int(item["prompt_count"])
        if pc not in keep_ids:
            continue
        if pc not in results_map:
            results_map[pc] = {"prompt_count": pc, "input": item.get("input", "")}

    # Optionally prefill from existing output_json (RESUME)
    prefilled_answers = 0
    resume_seen = set()
    if resume_json_path and Path(resume_json_path).exists():
        try:
            with open(resume_json_path, "r", encoding="utf-8") as rf:
                existing = json.load(rf)
            if isinstance(existing, list):
                for rec in existing:
                    if not isinstance(rec, dict) or "prompt_count" not in rec:
                        continue
                    pc = int(rec["prompt_count"])
                    resume_seen.add(pc)
                    if pc not in keep_ids:
                        # Different split/config; ignore those.
                        continue
                    if pc not in results_map:
                        results_map[pc] = {"prompt_count": pc, "input": rec.get("input", "")}
                    # Merge non-empty answers
                    for k, v in rec.items():
                        if k in {"prompt_count", "input"}:
                            if k == "input" and results_map[pc].get("input", "") == "":
                                results_map[pc]["input"] = v
                            continue
                        if isinstance(v, str) and v.strip():
                            if k not in results_map[pc] or not str(results_map[pc][k]).strip():
                                results_map[pc][k] = v
                                prefilled_answers += 1
            else:
                logging.warning("Existing output file is not a list; ignoring resume data.")
        except Exception as e:
            logging.warning("Failed to read/merge existing output_json (%s): %s", resume_json_path, e)

    # Build the generation queue of MISSING (or empty) keys
    flat_queue: List[Tuple[int, str, str, str]] = []
    missing_keys_counter: Dict[str, int] = {}

    # Normalise instruct_types once
    explicit_types = None
    if instruct_types:
        explicit_types = ["instruction_original"] + [k for k in instruct_types if k != "instruction_original"]

    held_groups = 0
    for item in raw_data:
        pc = int(item["prompt_count"])
        if pc not in keep_ids:
            continue
        held_groups += 1
        raw_input = item.get("input", "")

        if explicit_types:
            keep_keys = explicit_types
        else:
            keep_keys = ["instruction_original"] + [k for k in item.keys() if k.startswith("instruct_")]

        # De-duplicate order
        seen = set()
        keep_keys = [k for k in keep_keys if not (k in seen or seen.add(k))]

        if pc not in results_map:
            results_map[pc] = {"prompt_count": pc, "input": raw_input}

        for key in keep_keys:
            instr = item.get(key)
            if not instr:
                missing_keys_counter[key] = missing_keys_counter.get(key, 0) + 1
                continue
            existing_text = results_map[pc].get(key, "")
            if isinstance(existing_text, str) and existing_text.strip():
                continue
            flat_queue.append((pc, key, instr, raw_input))

    logging.info(
        "Loaded %d held-out groups (val∪test) | prompts to generate now: %d | prefilled answers: %d",
        held_groups, len(flat_queue), prefilled_answers
    )
    if resume_json_path and Path(resume_json_path).exists():
        extra = sorted(pc for pc in resume_seen if pc not in keep_ids)
        if extra:
            logging.warning("Existing output contained %d prompt_count IDs NOT in this run's held-out split (ignored). Example: %s",
                            len(extra), extra[:10])

    if missing_keys_counter:
        logging.warning("Some items were missing paraphrase keys (count by key): %s", missing_keys_counter)

    return flat_queue, results_map

# =========================
# Prompt formatting (MUST match training)
# =========================
def build_chat_prompt(tokenizer, instruction: str, inp: str = "") -> str:
    user_msg = instruction if not inp else f"{instruction}\n\nInput:\n{inp}"
    messages = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

# =========================
# ### ADDED: module access & swap for hybrids
# =========================
class BlockAccessor:
    def __init__(self, model: torch.nn.Module, layer_index: int):
        self.model = model
        # Common structures: LLaMA / Mistral / Gemma-ish
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            self.layers = model.transformer.h
        else:
            raise TypeError("Unsupported architecture (expected .model.layers or .transformer.h)")
        self.block = self.layers[layer_index]
        self.attn = getattr(self.block, "self_attn", None) or getattr(self.block, "attention", None)
        self.mlp  = getattr(self.block, "mlp", None) or getattr(self.block, "feed_forward", None)
        if self.attn is None or self.mlp is None:
            raise TypeError("Could not access attention or MLP submodule at the given layer.")

def _state_copy(dst: torch.nn.Module, src: torch.nn.Module):
    with torch.no_grad():
        for (n_d, p_d), (n_s, p_s) in zip(dst.named_parameters(), src.named_parameters()):
            if p_d.shape == p_s.shape:
                p_d.copy_(p_s)
        for (n_d, b_d), (n_s, b_s) in zip(dst.named_buffers(), src.named_buffers()):
            if b_d.shape == b_s.shape:
                b_d.copy_(b_s)

class SwapContext:
    """
    Temporarily swap submodules from FT into BASE at a given layer.
    Usage:
        with SwapContext(base, ft, L, swap_attn=True, swap_mlp=False) as hybrid:
            # run hybrid.generate(...)
    """
    def __init__(self, base: torch.nn.Module, ft: torch.nn.Module, layer_index: int, *, swap_attn: bool, swap_mlp: bool):
        self.base = base; self.ft = ft; self.layer_index = layer_index
        self.swap_attn = swap_attn; self.swap_mlp = swap_mlp
        self._backup_attn = None
        self._backup_mlp = None

    def __enter__(self):
        ba = BlockAccessor(self.base, self.layer_index)
        fa = BlockAccessor(self.ft,   self.layer_index)
        if self.swap_attn:
            self._backup_attn = ba.attn.state_dict()
            _state_copy(ba.attn, fa.attn)
        if self.swap_mlp:
            self._backup_mlp = ba.mlp.state_dict()
            _state_copy(ba.mlp, fa.mlp)
        return self.base

    def __exit__(self, exc_type, exc, tb):
        ba = BlockAccessor(self.base, self.layer_index)
        if self._backup_attn is not None:
            ba.attn.load_state_dict(self._backup_attn)
        if self._backup_mlp is not None:
            ba.mlp.load_state_dict(self._backup_mlp)
        self._backup_attn = None; self._backup_mlp = None
        return False

# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Held-out inference for paraphrx fine-tuning (with resume + causal patching hybrids)")

    # Dataset & split (unchanged)
    p.add_argument("--data_path", required=True, help="JSON file used in fine-tuning")
    p.add_argument("--instruct_types", nargs="+", default=[], help="Optional explicit list of instruct_* keys to use (default = ALL)")
    p.add_argument("--val_pct", type=float, default=0.05, help="Must match training")
    p.add_argument("--test_pct", type=float, default=0.05, help="Must match training")
    p.add_argument("--seed", type=int, default=42, help="Must match training")
    p.add_argument("--split", choices=["val", "test", "heldout"], default="heldout", help="'val', 'test', or both (heldout)")
    p.add_argument("--max_samples", type=int, default=0, help="Process at most this many prompt_count groups (0 = all)")
    p.add_argument("--from_prompt_id", type=int, default=0, help="Start processing from this prompt_count ID (inclusive)")
    p.add_argument("--upto_prompt_id", type=int, default=0, help="Process up to this prompt_count ID (inclusive, 0 = no limit)")

    # Models (BASE kept; FT ADDED)
    p.add_argument("--base_model_path", required=True, help="BASE model path")
    p.add_argument("--lora_path", help="(Legacy) LoRA adapter for BASE; usually leave empty")  # kept for full backward compat

    # ### ADDED: explicit FT inputs (preferred)
    p.add_argument("--ft_model_path", default=None, help="Fine-tuned MERGED model dir (preferred)")
    p.add_argument("--ft_lora_path", default=None, help="Fine-tuned LoRA adapter dir (will be merged if --ft_merge_lora)")
    p.add_argument("--ft_merge_lora", action="store_true", help="Merge FT LoRA into a base copy")

    p.add_argument("--merge_lora", action="store_true", help="(Legacy) merge adapter into base weights for faster inference")

    # Generation
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")

    # I/O (kept)
    p.add_argument("--output_json", required=True, help="Backward-compat single JSON path (used as FT file). Others will be written next to it.")
    p.add_argument("--outdir", default=None, help="Directory for all outputs; defaults to output_json's parent")
    p.add_argument("--wandb_project", default="paraphrx_50k_inf_ft")
    p.add_argument("--log_name", help="Unique name for this inference run (used in log filename and wandb)", default=None)
    p.add_argument("--save_every", type=int, default=100, help="Save answers every X batches")

    # ### ADDED: causal patching controls
    p.add_argument("--layer_index", type=int, default=6, help="Layer at which to patch ATT/MAT/MLP")
    p.add_argument("--run_base", action="store_true", help="Also generate with BASE (no FT).")
    p.add_argument("--run_ft", action="store_true", help="Also generate with FT (merged or LoRA).")
    p.add_argument("--run_hyb_attn", action="store_true", help="Generate with BASE + FT attention at --layer_index")
    p.add_argument("--run_hyb_mlp", action="store_true", help="Generate with BASE + FT MLP at --layer_index")
    p.add_argument("--run_hyb_both", action="store_true", help="Generate with BASE + FT attention+MLP at --layer_index")

    # If none of the above run_* flags are set, we'll run ALL five by default.
    return p.parse_args()

# =========================
# Main
# =========================
def main() -> None:
    args = parse_args()

    # Logging
    outdir = Path(args.outdir) if args.outdir else Path(args.output_json).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)

    Path("logs").mkdir(exist_ok=True)
    log_name = args.log_name or Path(args.base_model_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"infer_{log_name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    logging.info("CLI Args:\n%s", json.dumps(vars(args), indent=2))

    # Optional W&B
    if args.wandb_project:
        try:
            import wandb
            wb_run = wandb.init(
                project=args.wandb_project,
                name=f"infer_{log_name}",
                job_type="inference",
                config=vars(args),
            )
        except Exception as e:
            logging.warning("W&B init failed (%s) - continuing without logging", e)
            wb_run = None
    else:
        wb_run = None

    # Data (+ RESUME merge) — we’ll load RESUME **per-variant** so partial runs resume correctly.
    # Build the common flat_queue once (by asking for resume from the FT file path, just to get the queue).
    base_ft_path = Path(args.output_json)  # legacy path used as FT file
    variant_paths = {
        "BASE": outdir / "answers_BASE.json",
        "FT": base_ft_path,  # preserve your original path for FT variant
        "HYB_ATTN": outdir / "answers_HYB_ATTN.json",
        "HYB_MLP":  outdir / "answers_HYB_MLP.json",
        "HYB_BOTH": outdir / "answers_HYB_BOTH.json",
    }

    # What to run?
    flags = [args.run_base, args.run_ft, args.run_hyb_attn, args.run_hyb_mlp, args.run_hyb_both]
    if not any(flags):
        run_variants = ["BASE", "FT", "HYB_ATTN", "HYB_MLP", "HYB_BOTH"]
    else:
        run_variants = []
        if args.run_base: run_variants.append("BASE")
        if args.run_ft: run_variants.append("FT")
        if args.run_hyb_attn: run_variants.append("HYB_ATTN")
        if args.run_hyb_mlp: run_variants.append("HYB_MLP")
        if args.run_hyb_both: run_variants.append("HYB_BOTH")

    logging.info("Variants to run: %s", run_variants)

    # Load once to get the queue (no resume at this step)
    flat_queue_common, _ = load_examples_with_resume(
        args.data_path,
        instruct_types=args.instruct_types,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        seed=args.seed,
        split=args.split,
        max_samples=args.max_samples,
        from_prompt_id=args.from_prompt_id,
        upto_prompt_id=args.upto_prompt_id,
        resume_json_path=None,
    )
    if not flat_queue_common:
        logging.info("Nothing to do: no pending prompts for this split and filters.")
        # still write an empty FT file if needed
        for v in run_variants:
            if not variant_paths[v].exists():
                variant_paths[v].write_text("[]", encoding="utf-8")
        if wb_run: wb_run.finish()
        return

    # Sort shortest → longest to maximize batch utilisation
    flat_queue_common.sort(key=lambda t: len(t[2]))

    # For each variant, build its resume-aware results_map (and queue of STILL missing)
    per_variant_queue: Dict[str, List[Tuple[int,str,str,str]]] = {}
    per_variant_results: Dict[str, Dict[int, Dict]] = {}

    for variant in run_variants:
        q, res = load_examples_with_resume(
            args.data_path,
            instruct_types=args.instruct_types,
            val_pct=args.val_pct,
            test_pct=args.test_pct,
            seed=args.seed,
            split=args.split,
            max_samples=args.max_samples,
            from_prompt_id=args.from_prompt_id,
            upto_prompt_id=args.upto_prompt_id,
            resume_json_path=str(variant_paths[variant]),
        )
        # Ensure queue order matches common queue order (stable intersection)
        pending_set = {(pc, k) for (pc, k, _, _) in q}
        filtered = [t for t in flat_queue_common if (t[0], t[1]) in pending_set]
        per_variant_queue[variant] = filtered
        per_variant_results[variant] = res
        logging.info("Variant %s: pending=%d", variant, len(filtered))

    if wb_run:
        wb_run.config.update({"total_prompts_pending": len(flat_queue_common)}, allow_val_change=True)

    # =========================
    # Model & Tokeniser
    # =========================
    model_kwargs: dict = dict(device_map=args.device)

    # Flash Attention 2 (if available and not explicitly disabled)
    try:
        import importlib
        _FLASH2_OK = importlib.util.find_spec("flash_attn") is not None
    except Exception:
        _FLASH2_OK = False
    if os.getenv("DISABLE_FLASH_ATTN", "0") == "1":
        _FLASH2_OK = False
    if _FLASH2_OK:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        logging.info("Flash-Attention 2 not available - using standard attention")

    # Quantisation (bitsandbytes)
    try:
        import importlib
        _BNB_OK = importlib.util.find_spec("bitsandbytes") is not None
    except Exception:
        _BNB_OK = False
    if args.quant != "none" and not _BNB_OK:
        logging.warning("bitsandbytes not available - falling back to bf16")
        args.quant = "none"

    if args.quant == "none":
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=args.quant == "8bit",
            load_in_4bit=args.quant == "4bit",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    logging.info("Loading BASE model from %s", args.base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **model_kwargs).eval()

    # Build FT model (merged) — prefer explicit FT args
    ft_model: Optional[AutoModelForCausalLM] = None
    # Whether we need peft
    def _peft_ok() -> bool:
        try:
            import importlib
            return importlib.util.find_spec("peft") is not None
        except Exception:
            return False

    if args.ft_model_path:
        logging.info("Loading FT (merged) model from %s", args.ft_model_path)
        ft_model = AutoModelForCausalLM.from_pretrained(args.ft_model_path, **model_kwargs).eval()
    elif args.ft_lora_path:
        if not _peft_ok():
            logging.error("peft is not installed but --ft_lora_path provided")
            sys.exit(1)
        from peft import PeftModel
        logging.info("Loading FT as BASE+LoRA from %s", args.ft_lora_path)
        ft_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **model_kwargs)
        ft_model = PeftModel.from_pretrained(ft_model, args.ft_lora_path, is_trainable=False)
        if args.ft_merge_lora:
            logging.info("Merging FT LoRA into model …")
            ft_model = ft_model.merge_and_unload()
            logging.info("FT merge done")
        ft_model.eval()
    else:
        # Backward-compat fallback: use legacy args.lora_path & merge_lora AS the FT model
        if args.lora_path:
            if not _peft_ok():
                logging.error("peft is not installed but --lora_path provided")
                sys.exit(1)
            from peft import PeftModel
            logging.info("Legacy mode: using BASE+--lora_path as FT")
            ft_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **model_kwargs)
            ft_model = PeftModel.from_pretrained(ft_model, args.lora_path, is_trainable=False)
            if args.merge_lora:
                logging.info("Merging LoRA into FT …")
                ft_model = ft_model.merge_and_unload()
                logging.info("Merge done")
            ft_model.eval()
        else:
            logging.error("You must provide either --ft_model_path or --ft_lora_path (or legacy --lora_path).")
            sys.exit(1)

    # Try compile for slight speed
    try:
        base_model = torch.compile(base_model)
    except Exception:
        pass
    try:
        ft_model = torch.compile(ft_model)
    except Exception:
        pass

    # Tokenizer — prefer FT tokenizer if present, else BASE
    tok_path = (
        args.ft_model_path
        if args.ft_model_path and (Path(args.ft_model_path) / "tokenizer_config.json").exists()
        else args.base_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_path, model_max_length=4096)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =========================
    # Helpers for saving/resume & generation
    # =========================
    def _save_partial(variant: str):
        out_items = sorted(per_variant_results[variant].values(), key=lambda d: d["prompt_count"])
        variant_paths[variant].write_text(json.dumps(out_items, indent=2, ensure_ascii=False))
        if wb_run:
            try:
                import wandb
                wb_run.log({f"completed_groups_{variant}": len([d for d in out_items if len(d) > 2])})
            except Exception:
                pass

    def _handler(sig_num, _frame):
        logging.info("Signal %s caught - saving partial results for all variants", sig_num)
        for v in run_variants:
            _save_partial(v)
        sys.exit(0)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _handler)

    _INFER_CTX = getattr(torch, "inference_mode", torch.no_grad)

    SAVE_EVERY_N_BATCHES = args.save_every if args.save_every > 0 else (len(flat_queue_common) + args.batch - 1) // args.batch

    def to_device_inputs(tokeniser_outputs, device) -> dict:
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in tokeniser_outputs.items()}

    def generate_batch(model, tokenised, input_lens, max_new_tokens: int, temperature: float) -> List[str]:
        gen_cfg = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            gen_cfg["temperature"] = temperature
            gen_cfg["top_p"] = 0.6

        with _INFER_CTX():
            outputs = model.generate(**tokenised, **gen_cfg)

        out_texts = []
        for i in range(outputs.shape[0]):
            answer_ids = outputs[i, input_lens[i] :]
            text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            # Cleanup echo
            # (We can't easily reconstruct user message here—handled below in the caller)
            out_texts.append(text)
        return out_texts

    # =========================
    # Inference loop (single pass over the common queue; reuse tokenisation)
    # =========================
    logging.info("Starting inference. Total unique (prompt,key) pending across variants: %d", len(flat_queue_common))
    batches_done = 0

    # We'll accumulate a combined summary for plots
    summary_rows: List[Dict] = []

    # Precompute pending sets (avoid rebuilding every batch+variant)
    per_variant_pending_set: Dict[str, set[Tuple[int, str]]] = {
        v: {(pc, k) for (pc, k, _, _) in per_variant_queue[v]} for v in run_variants
    }

    for start in tqdm(range(0, len(flat_queue_common), args.batch), desc="generating"):
        batch = flat_queue_common[start : start + args.batch]
        pcs, keys, instrs, inputs = zip(*batch)
        prompts = [build_chat_prompt(tokenizer, i, inp) for i, inp in zip(instrs, inputs)]

        tokenised = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        input_lens = tokenised["attention_mask"].sum(dim=1)

        # PREP expected original-user strings for de-echo cleanup
        original_msgs = [instrs[i] if not inputs[i] else f"{instrs[i]}\n\nInput:\n{inputs[i]}" for i in range(len(batch))]

        def _gen_for_variant(variant_label: str, model_obj, swap_ctx=None):
            """Run generation only for rows pending in this variant."""
            if variant_label not in run_variants or len(per_variant_queue[variant_label]) == 0:
                return

            pend = per_variant_pending_set[variant_label]
            idx_list = [i for i in range(len(batch)) if (pcs[i], keys[i]) in pend]
            if not idx_list:
                return

            # Slice CPU tokenized batch to only needed rows
            sub_tokenised = {k: v[idx_list] if hasattr(v, "__getitem__") else v for k, v in tokenised.items()}
            sub_input_lens = input_lens[idx_list]

            # Move to the *current* model device
            sub_tokenised = to_device_inputs(sub_tokenised, model_obj.device)
            sub_input_lens = sub_input_lens.to(model_obj.device)

            # Generate
            texts = generate_batch(model_obj, sub_tokenised, sub_input_lens, args.max_tokens, args.temperature)

            # Map results back to global indices
            for local_i, global_i in enumerate(idx_list):
                text = texts[local_i]
                # Robust cleanup (echo)
                if text.strip() == original_msgs[global_i].strip():
                    logging.warning("%s: Empty generation for %s-%s (echo).", variant_label, pcs[global_i], keys[global_i])
                    text = ""
                else:
                    marker = "model\n"
                    pos = text.find(marker)
                    if pos != -1:
                        text = text[pos + len(marker):].lstrip()

                existing_text = per_variant_results[variant_label][pcs[global_i]].get(keys[global_i], "")
                if not (isinstance(existing_text, str) and existing_text.strip()):
                    per_variant_results[variant_label][pcs[global_i]][keys[global_i]] = text
                    summary_rows.append({
                        "variant": variant_label, "prompt_count": pcs[global_i], "key": keys[global_i],
                        "n_chars": len(text), "n_tokens_est": len(tokenizer.encode(text))
                    })

            logging.info("Batch %d: %s generated %d items", batches_done + 1, variant_label, len(idx_list))

        # 1) BASE
        if "BASE" in run_variants:
            _gen_for_variant("BASE", base_model)

        # 2) FT
        if "FT" in run_variants:
            _gen_for_variant("FT", ft_model)

        # 3) Hybrids (swap FT parts into BASE at layer_index, run only needed rows)
        def run_hybrid(label: str, swap_attn: bool, swap_mlp: bool):
            if label not in run_variants or len(per_variant_queue[label]) == 0:
                return
            with SwapContext(base_model, ft_model, args.layer_index, swap_attn=swap_attn, swap_mlp=swap_mlp) as hyb:
                _gen_for_variant(label, hyb)

        run_hybrid("HYB_ATTN", swap_attn=True,  swap_mlp=False)
        run_hybrid("HYB_MLP",  swap_attn=False, swap_mlp=True)
        run_hybrid("HYB_BOTH", swap_attn=True,  swap_mlp=True)

        # housekeeping
        del tokenised
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        batches_done += 1
        if (batches_done % SAVE_EVERY_N_BATCHES) == 0:
            logging.info("Saving partial results after %d batches (all variants that changed)", batches_done)
            for v in run_variants:
                _save_partial(v)

    # Final save
    for v in run_variants:
        _save_partial(v)
        logging.info("Finished %s → %s", v, variant_paths[v])

    # W&B artifact (optional)
    if wb_run:
        try:
            import wandb
            for v in run_variants:
                art = wandb.Artifact(
                    name=f"generations_{v}_{Path(args.base_model_path).stem}",
                    type="inference-results",
                    metadata={
                        "num_records": len(per_variant_results[v]),
                        "split": args.split,
                        "base_model": Path(args.base_model_path).name,
                        "variant": v,
                    },
                )
                art.add_file(str(variant_paths[v]))
                art.add_file(str(log_path))
                wb_run.log_artifact(art)
            wb_run.finish()
        except Exception as e:
            logging.warning("W&B artifact upload failed: %s", e)

    # =========================
    # ### ADDED: numeric summary + graphics
    # =========================
    import pandas as pd
    plots_dir = outdir / "plots"
    plots_dir.mkdir(exist_ok=True)

    summ_df = pd.DataFrame(summary_rows)
    if not summ_df.empty:
        summ_csv = outdir / "generations_summary_all_variants.csv"
        summ_df.to_csv(summ_csv, index=False)
        logging.info("Wrote summary CSV: %s (rows=%d)", summ_csv, len(summ_df))

        # (1) OVERLAID HISTOGRAM — token lengths
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        bins = 50
        for v in run_variants:
            vals = summ_df[summ_df["variant"] == v]["n_tokens_est"].values
            if len(vals) == 0: continue
            ax.hist(vals, bins=bins, histtype="step", linewidth=1.8, label=v)
        ax.set_title("Answer length (tokens) — overlaid")
        ax.set_xlabel("tokens (approx)")
        ax.set_ylabel("# generations")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "len_hist_overlaid.png", dpi=160)
        plt.close(fig)

        # (2) SCATTER — FT vs BASE token length (only where both exist)
        if "FT" in run_variants and "BASE" in run_variants:
            base_map = {(r["prompt_count"], r["key"]): r["n_tokens_est"] for _, r in summ_df[summ_df["variant"]=="BASE"].iterrows()}
            ft_map   = {(r["prompt_count"], r["key"]): r["n_tokens_est"] for _, r in summ_df[summ_df["variant"]=="FT"].iterrows()}
            xs, ys = [], []
            for k, v in ft_map.items():
                if k in base_map:
                    xs.append(base_map[k]); ys.append(v)
            if xs and ys:
                fig, ax = plt.subplots(figsize=(6,6))
                ax.scatter(xs, ys, s=6, alpha=0.5)
                lo = 0; hi = max(max(xs), max(ys)) + 5
                ax.plot([lo, hi], [lo, hi], linestyle="--")
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlabel("BASE tokens")
                ax.set_ylabel("FT tokens")
                ax.set_title("Answer length: FT vs BASE (per (prompt,key))")
                fig.tight_layout()
                fig.savefig(plots_dir / "len_scatter_FT_vs_BASE.png", dpi=160)
                plt.close(fig)

        # (3) SENSITIVITY — per-prompt dispersion across paraphrases
        #     For each variant, compute per-prompt std of n_tokens among its keys
        sens_rows = []
        for v in run_variants:
            vdf = summ_df[summ_df["variant"] == v]
            if vdf.empty: continue
            for pc, group in vdf.groupby("prompt_count"):
                std = float(np.std(group["n_tokens_est"].values, ddof=0))
                sens_rows.append({"variant": v, "prompt_count": pc, "sens_std_tokens": std})
        sens_df = pd.DataFrame(sens_rows)
        if not sens_df.empty:
            fig, ax = plt.subplots(figsize=(9,5))
            means = sens_df.groupby("variant")["sens_std_tokens"].mean().reindex(run_variants)
            ax.bar(means.index.tolist(), means.values.tolist())
            ax.set_title("Instruction sensitivity (per-prompt std of answer length)")
            ax.set_ylabel("std(tokens) across paraphrases")
            fig.tight_layout()
            fig.savefig(plots_dir / "sensitivity_bar.png", dpi=160)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9,5))
            sens_df.boxplot(column="sens_std_tokens", by="variant", ax=ax)
            ax.set_title("Instruction sensitivity distribution by variant")
            ax.set_ylabel("std(tokens) across paraphrases")
            plt.suptitle("")
            fig.tight_layout()
            fig.savefig(plots_dir / "sensitivity_box.png", dpi=160)
            plt.close(fig)

        # (4) CLOSENESS TO FT — |len(variant) − len(FT)|
        if "FT" in run_variants:
            ft_map = {(r["prompt_count"], r["key"]): r["n_tokens_est"] for _, r in summ_df[summ_df["variant"]=="FT"].iterrows()}
            close_rows = []
            for v in run_variants:
                if v == "FT": continue
                for _, r in summ_df[summ_df["variant"]==v].iterrows():
                    key = (int(r["prompt_count"]), r["key"])
                    if key in ft_map:
                        diff = abs(int(r["n_tokens_est"]) - int(ft_map[key]))
                        close_rows.append({"variant": v, "diff_to_FT": diff})
            close_df = pd.DataFrame(close_rows)
            if not close_df.empty:
                fig, ax = plt.subplots(figsize=(9,5))
                for v in [vv for vv in run_variants if vv != "FT"]:
                    vals = close_df[close_df["variant"]==v]["diff_to_FT"].values
                    if len(vals) == 0: continue
                    ax.hist(vals, bins=50, histtype="step", linewidth=1.8, label=v)
                ax.set_title("|length − FT length| — overlaid")
                ax.set_xlabel("absolute token diff")
                ax.set_ylabel("# generations")
                ax.legend()
                fig.tight_layout()
                fig.savefig(plots_dir / "closeness_to_FT_hist_overlaid.png", dpi=160)
                plt.close(fig)

                # Bars of mean closeness
                fig, ax = plt.subplots(figsize=(9,5))
                means = close_df.groupby("variant")["diff_to_FT"].mean().reindex([v for v in run_variants if v!="FT"])
                ax.bar(means.index.tolist(), means.values.tolist())
                ax.set_ylabel("mean |len - len_FT|")
                ax.set_title("Closeness to FT (lower is closer)")
                fig.tight_layout()
                fig.savefig(plots_dir / "closeness_to_FT_bar.png", dpi=160)
                plt.close(fig)

    # Final check for missing generations
    for v in run_variants:
        missing_any = [k for k, rec in per_variant_results[v].items() if len(rec) < 3]
        if missing_any:
            logging.warning("[%s] Some groups missing generations: %s", v, missing_any[:10])

    logging.info("All done. Outputs:")
    for v in run_variants:
        logging.info("  %s → %s", v, variant_paths[v])
    logging.info("  Summary CSV → %s", outdir / "generations_summary_all_variants.csv")
    logging.info("  Plots → %s", outdir / "plots")

if __name__ == "__main__":
    main()
