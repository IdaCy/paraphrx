#!/bin/bash -l
#SBATCH --job-name=l1-5_all_data_specific_layers_ft
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=11:00:00
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%x_%j_l1-5_all_data_specific_layers_ft.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%x_%j_l1-5_all_data_specific_layers_ft.err

set -euo pipefail
echo "$(date) – l1-5_all_data_specific_layers_ft job started on $(hostname)"
echo "$(date) – l1-5_all_data_specific_layers_ft job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

PROJ=/scratch_root/ifc24/paraphrx
cd "$PROJ"
mkdir -p logs tmp hf_cache pip_cache xdg_cache f_finetune/outputs/all_data_specific_layers/outputs_buckets_1-5
export TMPDIR=$PROJ/tmp
export HF_HOME=$PROJ/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export PIP_CACHE_DIR=$PROJ/pip_cache
export XDG_CACHE_HOME=$PROJ/xdg_cache
export PYTHONNOUSERSITE=1
export PYTHONPATH=
export HOME=$TMPDIR
export PATH=$HOME/.local/bin:$PATH

# Pick an available Python-3 binary (prefers 3.11 - as 3.8)
for P in python3.11 python3.10 python3.9 python3; do
  if command -v $P >/dev/null 2>&1; then PYBIN=$(command -v $P); break; fi
done
[ -z "${PYBIN:-}" ] && { echo "No python3 on PATH"; exit 1; }
echo "Using $($PYBIN -V)"

# install standalone Miniconda in TMPDIR and activate
CONDADIR="$TMPDIR/conda"
ENVDIR="$TMPDIR/conda_env"
if [ ! -f "$CONDADIR/bin/conda" ]; then
  wget -qO "$TMPDIR/miniconda.sh" \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$TMPDIR/miniconda.sh" -b -p "$CONDADIR"
  rm "$TMPDIR/miniconda.sh"
fi
export PATH="$CONDADIR/bin:$PATH"

# auto-accept Terms of Service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# create a fresh env with just Python and pip
conda create -y -q -p "$ENVDIR" python=3.10 pip
source "$CONDADIR/bin/activate" "$ENVDIR"

# upgrade pip/setuptools/wheel
pip install --upgrade pip setuptools wheel -q

# **Install PyTorch GPU wheel via pip** (replacing the conda build that was missing symbols)
pip install --prefer-binary \
  torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# install the rest of your stack (including bitsandbytes ≥0.43.2)
pip install --prefer-binary \
  transformers==4.45.0 \
  peft==0.11.1 \
  bitsandbytes>=0.43.2 \
  mpi4py \
  deepspeed \
  datasets==2.19.0 \
  numpy==1.24.4 sentencepiece loguru protobuf==3.20.* wandb testresources matplotlib -q
pip uninstall -y torchaudio torchvision -q || true

python - <<'PY'
import torch, transformers, pathlib, importlib.util, numpy as np
print("Torch       :", torch.__version__)
print("CUDA?       :", torch.cuda.is_available(), torch.version.cuda)
print("Bits&Bytes  :", __import__("bitsandbytes").__version__)
print("Transformers:", transformers.__version__)
print("NumPy       :", np.__version__)
print("Loaded from :", pathlib.Path(importlib.util.find_spec('transformers').origin).parent)
PY

source /scratch_root/ifc24/.hf_token || true
[ -f /scratch_root/ifc24/.wandb_key ] && source /scratch_root/ifc24/.wandb_key
WANDB_FLAG=${WANDB_API_KEY:+--wandb_project paraphrx_lora}

export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

RUN_SCRIPT="f_finetune/src/finetuning_specific_layers.py"
srun python "$RUN_SCRIPT" \
  --data_paths alpaca_gemma-2-2b-it \
  --model_path f_finetune/model \
  --output_dir "f_finetune/outputs/alpaca_all_layers_XXXZZ/outputs_buckets_1-2" \
  --run_name buckets_1-2_all_layers \
  --buckets 1-2 \
  --bf16 \
  --bnb_8bit_optim \
  --target_modules XXXXXXZZ \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 8e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --num_epochs 3 \
  --save_steps 200 \
  $WANDB_FLAG

echo "$(date) – l1-5_all_data_specific_layers_ft job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
echo "$(date) – l1-5_all_data_specific_layers_ft job finished OK"
