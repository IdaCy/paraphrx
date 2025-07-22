#!/bin/bash -l
#SBATCH --job-name=lA-base_1-3_a_all_layers_ft
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_lA-base_1-3_a_all_layers_ft.out
#SBATCH --error=/scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_lA-base_1-3_a_all_layers_ft.err

set -euo pipefail
echo "$(date) – lA-base_1-3_a_all_layers_ft job started on $(hostname)"
echo "$(date) – lA-base_1-3_a_all_layers_ft job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_dgxl/ifc24/proj/paraphrx
cd "$PROJ"
mkdir -p logs f_finetune/outputs_5/alpaca_all_layers/outputs_buckets_1-3_A-base

export TMPDIR=$PROJ/tmp_${SLURM_JOB_ID}
export HF_HOME=$PROJ/hf_cache_${SLURM_JOB_ID}
export PIP_CACHE_DIR=$PROJ/pip_cache_${SLURM_JOB_ID}
export XDG_CACHE_HOME=$PROJ/xdg_cache_${SLURM_JOB_ID}

export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export PYTHONNOUSERSITE=1
export PYTHONPATH=
export HOME=$TMPDIR
export PATH=$HOME/.local/bin:$PATH

cleanup () {
  echo "$(date) – cleaning scratch for $SLURM_JOB_ID"
  rm -rf "$TMPDIR" "$HF_HOME" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"
}
# run cleanup on:
#   EXIT  – normal completion
#   ERR   – any command that triggers `set -e`
#   INT   – Ctrl‑C from interactive dev
#   TERM  – scancel / pre‑emption
trap cleanup EXIT ERR INT TERM

mkdir -p "$TMPDIR" "$HF_HOME" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"

# Pick an available Python-3 binary (prefers 3.11 - as 3.8)
for P in python3.11 python3.10 python3.9 python3; do
  if command -v $P >/dev/null 2>&1; then PYBIN=$(command -v $P); break; fi
done
[ -z "${PYBIN:-}" ] && { echo "No python3 on PATH"; exit 1; }
echo "Using $($PYBIN -V)"

# create fresh venv in node-local scratch
VENV_DIR="$TMPDIR/venv_$SLURM_JOB_ID"
rm -rf "$VENV_DIR"
$PYBIN -m pip install --user virtualenv -q
virtualenv -p "$PYBIN" "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel -q

python -m pip install --prefer-binary \
  torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121 \
  transformers==4.45.0 \
  peft==0.11.1 \
  bitsandbytes==0.42.0 \
  mpi4py \
  deepspeed \
  datasets==2.19.0 \
  numpy==1.24.4 sentencepiece loguru protobuf==3.20.* wandb testresources matplotlib -q
python -m pip uninstall -y torchaudio torchvision -q || true

python - <<'PY'
import torch, transformers, pathlib, importlib.util, numpy as np
print("Torch       :", torch.__version__)
print("CUDA?       :", torch.cuda.is_available(), torch.version.cuda)
print("Bits&Bytes  :", __import__("bitsandbytes").__version__)
print("Transformers:", transformers.__version__)
print("NumPy       :", np.__version__)
print("Loaded from :", pathlib.Path(importlib.util.find_spec('transformers').origin).parent)
PY

source /scratch_dgxl/ifc24/.hf_token || true
[ -f /scratch_dgxl/ifc24/.wandb_key ] && source /scratch_dgxl/ifc24/.wandb_key
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrx_lora}

# silence tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# headless plotting
export MPLBACKEND=Agg

RUN_SCRIPT="f_finetune/src/finetuning.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs_5/alpaca_all_layers/outputs_buckets_1-3_A-base" \
  --run_name A-base_buckets_1-3_all_layers \
  --buckets 1-3 \
  --bf16 \
  --target_modules none \
  --batch_size 4 \
  --gradient_accumulation_steps 16 \
  --learning_rate 3e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --eval_steps 200 \
  --save_steps 200 \
  $WANDB_FLAG

echo "$(date) – lA-base_1-3_a_all_layers_ft job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) – lA-base_1-3_a_all_layers_ft job finished OK"
