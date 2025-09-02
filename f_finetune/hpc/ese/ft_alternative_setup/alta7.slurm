#!/bin/bash -l
#SBATCH --job-name=hlalta7
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch_dgxl/ifc24/proj/paraphrx/logs/%j_%x.out
#SBATCH --error=/scratch_dgxl/ifc24/proj/paraphrx/logs/%j_%x.err

set -euo pipefail
echo "$(date) - hlalta7 job started on $(hostname)"
echo "$(date) - hlalta7 job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_dgxl/ifc24/proj/paraphrx
cd "$PROJ"
mkdir -p logs f_finetune/outputs_lap_pr/hlalta7

export TMPDIR="${SLURM_LOCALTMP:-/tmp}"

# Shared HF cache
export HF_HOME="$PROJ/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME"

# Redirect Triton autotune cache
export TRITON_CACHE_DIR="$TMPDIR/triton_cache_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"

export PIP_CACHE_DIR="$TMPDIR/pip_cache_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="$TMPDIR/xdg_cache_${SLURM_JOB_ID}"
export WANDB_DIR="$TMPDIR/wandb_cache_${SLURM_JOB_ID}"
export WANDB_DATA_DIR="$TMPDIR/wandb_data_${SLURM_JOB_ID}"
export PYTHONNOUSERSITE=1
export PYTHONPATH=
export PATH=$HOME/.local/bin:$PATH

cleanup () {
  echo "$(date) - cleaning up only my directories for $SLURM_JOB_ID"
  rm -rf \
    "$PIP_CACHE_DIR" \
    "$XDG_CACHE_HOME" \
    "$WANDB_DIR" \
    "$TRITON_CACHE_DIR"
}
trap cleanup EXIT ERR INT TERM

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_DIR"

# Pick an available Python-3 binary (prefers 3.11 → 3.8)
for P in python3.11 python3.10 python3.9 python3; do
  if command -v $P &>/dev/null; then PYBIN=$(command -v $P); break; fi
done
[ -z "${PYBIN:-}" ] && { echo "No python3 on PATH"; exit 1; }
echo "Using $($PYBIN -V)"

export PYTHON_PKGS="$TMPDIR/python-packages"
# --- NEW: Force a clean directory for packages ---
echo "Forcing clean package directory at $PYTHON_PKGS"
rm -rf "$PYTHON_PKGS"
mkdir -p "$PYTHON_PKGS"
# --- END NEW ---
export PYTHONPATH="$PYTHON_PKGS:$PYTHONPATH"

# DEEP-SEED compatible metadata libs
$PYBIN -m pip install --upgrade "importlib_metadata>=5.2.0" \
                                   "setuptools==65.5.1" \
                                   "testresources==2.0.1" \
    --no-cache-dir --target "$PYTHON_PKGS"

# install pip & wheel (keep your pinned setuptools)
$PYBIN -m pip install --upgrade pip wheel \
  --no-cache-dir --target "$PYTHON_PKGS"

# deepspeed to a wheelable version!
$PYBIN -m pip install --prefer-binary \
  torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121 \
  transformers==4.45.0 \
  peft==0.11.1 \
  "bitsandbytes>=0.43.2" \
  mpi4py \
  datasets==2.19.0 \
  numpy==1.24.4 \
  sentencepiece \
  loguru \
  protobuf==3.20.* \
  wandb \
  matplotlib \
  --no-cache-dir \
  --target "$PYTHON_PKGS"

$PYBIN - <<'PY'
import torch, transformers, pathlib, importlib.util, numpy as np
print("Torch       :", torch.__version__)
print("CUDA?       :", torch.cuda.is_available(), torch.version.cuda)
print("Bits&Bytes  :", __import__("bitsandbytes").__version__)
print("Transformers:", transformers.__version__)
print("NumPy       :", np.__version__)
print("Loaded from :", pathlib.Path(importlib.util.find_spec('transformers').origin).parent)
PY

source /scratch_dgxl/ifc24/.hf_token || true
[ -f /scratch_dgxl/ifc24/.wandb_idacasp_key ] && source /scratch_dgxl/ifc24/.wandb_idacasp_key
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrx_ft_50k}

export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

echo "$(date) - hlalta7 initial setup OK"
RUN_SCRIPT="f_finetune/src/ft_alternative_setup.py"
OUTPUT_DIR="f_finetune/outputs_alternat/hlalta7"
MODEL_PATH="f_finetune/model"

WANDB_PROJECT="paraphrx_ft_lap"
RUN_NAME="hlalta7"

# sbatch --dependency=afterany:1799:1794 f_finetune/hpc/borg/borg_ft_lap_9xreal_stable5.slurm

srun "$PYBIN" "$RUN_SCRIPT" \
  --data_paths e_eval/data/alpaca_gemma-2-2b-it.json e_eval/data/gsm8k_gemma-2-2b-it.json e_eval/data/mmlu_gemma-2-2b-it.json a_data/alpaca/50k_phrxed.json \
  --require_content_score 4 --first_min_style_count 200 \
  --curriculum_style_jsons a_data/alpaca/selections/selection_useable_s.json a_data/alpaca/selections/selection_useable_m.json \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --run_name alta7_l8_attn_adv \
  --curriculum_stage_epochs 2 2 1 \
  --target_modules q_proj,k_proj,v_proj,o_proj \
  --lora_layer_idx 8 --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 \
  --batch_size 4 --gradient_accumulation_steps 8 \
  --num_epochs 5 --learning_rate 2e-5 --warmup_ratio 0.05 \
  --consistency_kld_weight 0.40 \
  --contrast_weight 0.00 \
  --style_adv_weight 0.12 --style_adv_hidden 256 --style_adv_lambda 1.0 \
  --use_lap --lap_layer 8 --lap_t_inner 3 --lap_p_sample 0.30 --lap_epsilon 0.04 \
  --bf16 --bnb_8bit_optim \
  --eval_steps 400 --save_steps 400 --save_total_limit 30 \
  --model_only_checkpoints --early_stopping_patience 3 \
  --logging_steps 50 --seed 42

echo "$(date) - hlalta7 job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) - hlalta7 job finished OK"
