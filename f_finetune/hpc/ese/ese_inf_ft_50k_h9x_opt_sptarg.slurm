#!/bin/bash -l
#SBATCH --job-name=li9x_opt_sptarg
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_li9x_opt_sptarg.out
#SBATCH --error=/scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_li9x_opt_sptarg.err

set -euo pipefail
echo "$(date) - li9x_opt_sptarg job started on $(hostname)"
echo "$(date) - li9x_opt_sptarg job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_dgxl/ifc24/proj/paraphrx
cd "$PROJ"
mkdir -p logs f_finetune/output_inf_ft_50k

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
mkdir -p "$PYTHON_PKGS"
export PYTHONPATH="$PYTHON_PKGS:$PYTHONPATH"

# install pip & wheel (keep your pinned setuptools)
$PYBIN -m pip install --upgrade pip wheel \
  --no-cache-dir --target "$PYTHON_PKGS"

# deepspeed to a wheelable version!
$PYBIN -m pip install --prefer-binary \
  torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121 \
  transformers==4.45.0 \
  peft==0.11.1 \
  datasets==2.19.0 \
  numpy==1.24.4 \
  wandb \
  --no-cache-dir \
  --target "$PYTHON_PKGS"

$PYBIN - <<'PY'
import torch, transformers, pathlib, importlib.util, numpy as np
print("Torch       :", torch.__version__)
print("CUDA?       :", torch.cuda.is_available(), torch.version.cuda)
print("Transformers:", transformers.__version__)
print("NumPy       :", np.__version__)
print("Loaded from :", pathlib.Path(importlib.util.find_spec('transformers').origin).parent)
PY

source /scratch_dgxl/ifc24/.hf_token || true
[ -f /scratch_dgxl/ifc24/.wandb_idacasp_key ] && source /scratch_dgxl/ifc24/.wandb_idacasp_key
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrx_ft_50k}

export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

RUN_SCRIPT="f_finetune/src/inf_ft_50k.py"
srun $PYBIN "$RUN_SCRIPT" \
  --data_path a_data/alpaca/50k_phrxed.json \
  --instruct_types \
    instruct_formal_memo \
    instruct_joke \
    instruct_sardonic \
    instruct_future_tense \
    instruct_coord_to_subord \
    instruct_dramatic \
    instruct_polite_request \
    instruct_one_typo_punctuation \
  --base_model_path f_finetune/model \
  --lora_path f_finetune/outputs/h9x_opt_sptarg_50k_ft/final \
  --output_json f_finetune/output_inf_ft_50k/li9x_opt_sptarg.json \
  --log_name li9x_opt_sptarg \
  --val_pct 0.05 --test_pct 0.05 \
  --batch 8 --max_tokens 128 \
  --quant 4bit \
  --wandb_project paraphrx_inference \
  --max_samples 500 \
  $WANDB_FLAG

echo "$(date) - li9x_opt_sptarg job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) - li9x_opt_sptarg job finished OK"
