#!/bin/bash -l
#SBATCH --job-name=s1_sft_consistency
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_s1_sft_consistency.out
#SBATCH --error=/scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_s1_sft_consistency.err

set -euo pipefail
echo "$(date) - s1_sft_consistency job started on $(hostname)"
echo "$(date) - s1_sft_consistency job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_dgxl/ifc24/proj/paraphrx
cd "$PROJ"
mkdir -p logs f_finetune/outputs_ft_2/pipetwo
mkdir -p f_finetune/outputs_ft_2/pipetwo/s2_orpo
mkdir -p f_finetune/outputs_ft_2/pipetwo/s2_groupdro

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
  deepspeed \
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

RUN_SCRIPT="f_finetune/src/ft_2.py"
BASE_MODEL="f_finetune/model"

echo "$(date) - stage 1 starting"

PROMPTS_SECOND="a_data/alpaca/50k_phrxed_nofirst500.json"
PROMPTS_FIRST="a_data/alpaca/paraphrases_500.json"
STYLES_SECOND="a_data/alpaca/selections/50k_phrxed_nofirst500.json"
STYLES_FIRST="a_data/alpaca/selections/paraphrases_500.json"
ANSWERS_SECOND="c_assess_inf/output50k/answers.json"

echo "$(date) - Stage 1 (SFT) starting"

srun $PYBIN "$RUN_SCRIPT" \
  --data_paths "$PROMPTS_SECOND" "$PROMPTS_FIRST" \
  --styles_paths "$STYLES_SECOND" "$STYLES_FIRST" \
  --model_path "$BASE_MODEL" \
  --output_dir f_finetune/outputs_ft_2/pipetwo/s2_orpo \
  --run_name s2_orpo \
  --bf16 \
  --orpo_beta 0.1 \
  --batch_size 8 --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 --num_epochs 1 \
  $WANDB_FLAG

# Get best checkpoint dir (adjust if you know the ID)
CKPT_DIR=$(ls -d f_finetune/outputs_ft_2/pipetwo/s2_orpo/checkpoint-* | sort -V | tail -n1)

echo "$(date) - Stage 1.5 (GroupDRO) starting"

srun $PYBIN "$RUN_SCRIPT" \
  --data_paths "$PROMPTS_SECOND" "$PROMPTS_FIRST" \
  --styles_paths "$STYLES_SECOND" "$STYLES_FIRST" \
  --model_path "$BASE_MODEL" \
  --resume_checkpoint "$CKPT_DIR" \
  --output_dir f_finetune/outputs_ft_2/pipetwo/s2_groupdro \
  --run_name s2_groupdro \
  --bf16 \
  --use_groupdro --groupdro_eta 0.05 \
  --batch_size 8 --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 --num_epochs 1 \
  --wandb_project paraphrx_best2
  $WANDB_FLAG

# Optional Stage 2 (ORPO) — needs answers JSON and only ONE prompts JSON (SECOND)
echo "$(date) - Stage 2 (ORPO) starting"

echo "$(date) - s1_sft_consistency job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) - s1_sft_consistency job finished OK"
