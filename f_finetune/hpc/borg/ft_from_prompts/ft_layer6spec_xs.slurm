#!/bin/bash -l
#SBATCH --job-name=xsft_spec6layer
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%x_%j_ft_6layer.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%x_%j_ft_6layer.err

set -euo pipefail
echo "$(date) - xsft_spec6layer job started on $(hostname)"
echo "$(date) - xsft_spec6layer job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_root/ifc24/paraphrx
cd "$PROJ"

mkdir -p logs f_finetune/outputs_great_nolap/xsft_spec6layer

export SCRATCH=/scratch_root/ifc24
export TMPDIR="$SCRATCH/tmp/${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
echo "Using scratch-backed TMPDIR=$TMPDIR"

export HF_HOME="$PROJ/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME"
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
  echo "$(date) - cleaning up ENTIRE temp directory for $SLURM_JOB_ID: $TMPDIR"
  rm -rf "$TMPDIR"
}
trap cleanup EXIT ERR INT TERM

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_DIR"

for P in python3.11 python3.10 python3.9 python3; do
  if command -v $P &>/dev/null; then PYBIN=$(command -v $P); break; fi
done
[ -z "${PYBIN:-}" ] && { echo "No python3 on PATH"; exit 1; }
echo "Using $($PYBIN -V)"
export PYTHON_PKGS="$TMPDIR/python-packages"
mkdir -p "$PYTHON_PKGS"
export PYTHONPATH="$PYTHON_PKGS:$PYTHONPATH"
curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$TMPDIR/get-pip.py"
$PYBIN "$TMPDIR/get-pip.py" --target "$PYTHON_PKGS" --no-cache-dir
$PYBIN -m pip install --upgrade "importlib_metadata>=5.2.0" "testresources==2.0.1" --no-cache-dir --target "$PYTHON_PKGS"
$PYBIN -m pip install --upgrade pip wheel --no-cache-dir --target "$PYTHON_PKGS"
$PYBIN -m pip install --upgrade setuptools --no-cache-dir --target "$PYTHON_PKGS"
$PYBIN -m pip install --prefer-binary --only-binary=:all: \
  torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121 \
  transformers==4.45.0 \
  safetensors==0.4.3 \
  peft==0.11.1 \
  bitsandbytes==0.43.2 \
  mpi4py \
  datasets==2.19.0 \
  numpy==1.26.4 \
  sentencepiece \
  loguru \
  protobuf==3.20.* \
  wandb \
  matplotlib \
  --no-cache-dir \
  --target "$PYTHON_PKGS"
$PYBIN - <<'PY'
import torch, transformers, pathlib, importlib.util, numpy as np, safetensors
print("Torch       :", torch.__version__)
print("CUDA?       :", torch.cuda.is_available(), torch.version.cuda)
print("Bits&Bytes  :", __import__("bitsandbytes").__version__)
print("Transformers:", transformers.__version__)
print("Safetensors :", safetensors.__version__)
print("NumPy       :", np.__version__)
print("Loaded from :", pathlib.Path(importlib.util.find_spec('transformers').origin).parent)
PY
source /scratch_root/ifc24/.hf_token || true
[ -f /scratch_root/ifc24/.wandb_idacasp_key ] && source /scratch_root/ifc24/.wandb_idacasp_key
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrx_ft_50k}
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

echo "$(date) - xsft_spec6layer initial setup OK"
RUN_SCRIPT="f_finetune/src/finetuning_50k_resume_any_aw.py"

MODEL_PATH="f_finetune/model"
OUTPUT_DIR="f_finetune/outputs_great_nolap/xsft_spec6layer"
RUN_NAME="xsft_spec6layer"

# sbatch --dependency=afterany:1799:1794 f_finetune/hpc/borg/borg_ft_lap_9xreal_stable5.slurm

srun $PYBIN "$RUN_SCRIPT" \
  --data_paths a_data/alpaca/50k_phrxed.json \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --run_name xsft_spec6layer \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_layer_idx 6 \
  --lora_rank 64 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_epochs 3 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine_with_restarts \
  --weight_decay 0.0 \
  --bnb_8bit_optim --bf16 \
  --eval_steps 500 --save_steps 500 \
  --early_stopping_patience 3 \
  --logging_steps 100 \
  $WANDB_FLAG

echo "$(date) - xsft_spec6layer job itself finished OK"

echo "$(date) - xsft_spec6layer job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) - xsft_spec6layer job finished & stuff cleaned OK"
