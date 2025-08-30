#!/bin/bash -l
#SBATCH --job-name=L2ac_caus1
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%j_%x.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%j_%x.err

set -euo pipefail
echo "$(date) - L2ac_caus1 job started on $(hostname)"
echo "$(date) - L2ac_caus1 job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_root/ifc24/paraphrx
cd "$PROJ"

mkdir -p logs

export SCRATCH=/scratch_root/ifc24
export TMPDIR="$SCRATCH/tmp/${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
echo "Using scratch-backed TMPDIR=$TMPDIR"

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
  echo "$(date) - cleaning up for $SLURM_JOB_ID (TMPDIR=$TMPDIR)"
  if [[ -n "${TMPDIR:-}" && "$TMPDIR" == "$SCRATCH/tmp/"* ]]; then
    rm -rf "$TMPDIR"
  fi
  if [[ -n "${HF_HOME:-}" && "$HF_HOME" == "$PROJ/hf_cache" ]]; then
    rm -rf "$HF_HOME"
  fi
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

curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$TMPDIR/get-pip.py"

# Install pip into your PYTHON_PKGS target area
$PYBIN "$TMPDIR/get-pip.py" --target "$PYTHON_PKGS" --no-cache-dir

# install pip & wheel (keep your pinned setuptools)
$PYBIN -m pip install --upgrade pip wheel \
  --no-cache-dir --target "$PYTHON_PKGS"

$PYBIN -m pip install --upgrade setuptools \
  --no-cache-dir --target "$PYTHON_PKGS"

# install everything from wheels only (no source builds)
$PYBIN -m pip install --prefer-binary --only-binary=:all: \
  torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121 \
  transformers==4.45.0 \
  peft==0.11.1 \
  bitsandbytes==0.43.2 \
  datasets==2.19.0 \
  numpy==1.26.4 \
  pandas \
  scipy \
  seaborn \
  tqdm \
  wandb \
  matplotlib \
  jsonlines \
  scikit-learn \
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

source /scratch_root/ifc24/.hf_token || true
[ -f /scratch_root/ifc24/.wandb_idacasp_key ] && source /scratch_root/ifc24/.wandb_idacasp_key
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrx_lora}

export TOKENIZERS_PARALLELISM=false

RUN_SCRIPT="f_finetune/src/ft_inference_all_data.py"
BASE_MODEL_PATH="f_finetune/model"
FT_PATH="f_finetune/outputs_6/all_data_all_layers/1-2_lowLR_fullFT"
OUTPUT_JSON="$FT_PATH/inf_results.json"
LORA_PATH="$FT_PATH/final"
mkdir -p "$(dirname "$OUTPUT_JSON")"

# sbatch --dependency=afternotok:1143 c_assess_inf/hpc/borg_inf_acts.slurm

# Run
echo "$(date) - L2ac_caus1 initial setup OK"
RUN_SCRIPT="f_finetune/src/analyses_2_preneurips/ac_caus.py"

# ATTN + MLP candidates; small K; still 'last' slice to avoid mean-shift quirks
DATA_PATH="f_finetune/data/first_dataset_sampler/jacobian_prompts.jsonl"
BASE_MODEL_PATH="f_finetune/model"
FT_MODEL_PATH="f_finetune/outputs_great_nolap/ft_spec6layer/final"
OUTPUT_STEM="f_finetune/outputs_great_nolap/ft_spec6layer/analyses_2_preneurips/"

# Run
echo "$(date) - Aac_caus1 initial setup OK"
RUN_SCRIPT="f_finetune/src/analyses_2_preneurips/ac_caus.py"
srun $PYBIN "$RUN_SCRIPT" \
  --data_path "$DATA_PATH" \
  --base_model_path "$BASE_MODEL_PATH" \
  --ft_model_path "$FT_MODEL_PATH" \
  --ft_is_lora --merge_lora \
  --output_dir "$OUTPUT_STEM/ac_cause_L6_paper_nec_headline_L6pair_K2_fixed" \
  --val_pct 0.05 --test_pct 0.05 --seed 42 \
  --groupwise_split \
  --holdout_para_frac 0.33 \
  --max_train_pairs 5000 --max_eval_pairs 0 \
  --batch_size 64 \
  --layers 6,10,14,18 \
  --modules attn_out,mlp_post \
  --slices last \
  --iit_steps 1200 --iit_lr 4e-2 --lmbd_sparse 1.5e-2 \
  --temp_start 1.2 --temp_end 0.25 \
  --ladder_start_K 2 --ladder_final_K 2 \
  --quant none


echo "$(date) - L2ac_caus1 job itself finished OK"

rm -rf "$TMPDIR" \
       "$HF_HOME" \
       "$PIP_CACHE_DIR" \
       "$XDG_CACHE_HOME"

echo "$(date) - L2ac_caus1 job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) - L2ac_caus1 job finished & stuff cleaned OK"
