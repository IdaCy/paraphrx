#!/bin/bash -l
#SBATCH --job-name=alta13lap
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%j_%x.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%j_%x.err

set -euo pipefail
echo "$(date) - alta13lap job started on $(hostname)"
echo "$(date) - alta13lap job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_root/ifc24/paraphrx
cd "$PROJ"

mkdir -p logs f_finetune/outputs_alternat/alta13lap

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
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrax_ft_nolap}
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

echo "$(date) - alta13lap initial setup OK"
RUN_SCRIPT="f_finetune/src/ft_alternative_setup.py"
OUTPUT_DIR="f_finetune/outputs_alternat/alta13lap"
MODEL_PATH="f_finetune/model"

WANDB_PROJECT="paraphrx_ft_lap"
RUN_NAME="alta13lap"

# sbatch --dependency=afterany:1799:1794 f_finetune/hpc/borg/borg_ft_lap_9xreal_stable5.slurm

srun "$PYBIN" "$RUN_SCRIPT" \
  --data_paths e_eval/data/alpaca_gemma-2-2b-it.json e_eval/data/gsm8k_gemma-2-2b-it.json e_eval/data/mmlu_gemma-2-2b-it.json a_data/alpaca/50k_phrxed.json \
  --model_path f_finetune/model \
  --output_dir "$OUTPUT_DIR" \
  --batch_size 2 --gradient_accumulation_steps 16 \
  --num_epochs 3 --learning_rate 3e-5 --warmup_ratio 0.03 \
  --bf16 \
  --eval_steps 700 --save_steps 700 --save_total_limit 5 \
  --model_only_checkpoints --early_stopping_patience 6 \
  --logging_steps 50 --seed 42 \
  \
  --curriculum_style_jsons a_data/alpaca/selections/only_original.json \
  \
  --use_lap \
  --lap_p_sample 1.0 \
  --lap_layer 12 \
  --lap_t_inner 3 \
  --lap_epsilon 0.05 \
  --lap_delta_lr 1e-2 \
  --lap_lambda_lr 1e-3 \
  \
  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  \
  --consistency_kld_weight 0.0 \
  --contrast_weight 0.0 \
  --style_adv_weight 0.0

echo "$(date) - alta13lap job itself finished OK"

echo "$(date) - alta13lap job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) - alta13lap job finished & stuff cleaned OK"
