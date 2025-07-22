#!/bin/bash -l
#SBATCH --job-name=ft_stats
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:05:00
#SBATCH -o /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_ft_stats.out
#SBATCH -e /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_ft_stats.err
set -euo pipefail

echo "$(date) - ft_stats job started on $(hostname)"
echo "$(date) - ft_stats job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

# Workspace + caches on scratch
cd /scratch_dgxl/ifc24/proj/paraphrx
mkdir -p logs tmp hf_cache pip_cache xdg_cache

export TMPDIR=$PWD/tmp
export HF_HOME=$PWD/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export PIP_CACHE_DIR=$PWD/pip_cache
export XDG_CACHE_HOME=$PWD/xdg_cache
export TORCH_HOME=$XDG_CACHE_HOME/torch
export TORCHINDUCTOR_CACHE_DIR=$TORCH_HOME/inductor
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
mkdir -p "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Python environment
if [ ! -d venv ]; then
    echo "Creating venv ..."
    python3.12 -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip --quiet
python -m pip install numpy pandas matplotlib scikit-learn \
          torch torchvision torchaudio \
                       --index-url https://download.pytorch.org/whl/cu121 --quiet
python -m pip install "transformers>=4.40.0" accelerate tqdm --quiet
python -m pip install huggingface_hub sentencepiece "protobuf==3.20.*" hf_transfer --quiet
echo "Packages ready."


python3 e_eval/src/tf_stats.py \
    --orig_json f_finetune/data/all_alpaca_gemma-2-2b-it_all_ppl.json \
    --finetune_jsons \
    f_finetune/outputs/1st_run/alpaca/all_layers/ft_inf_scores/buckets_1-1.json \
    f_finetune/outputs/1st_run/alpaca/all_layers/ft_inf_scores/buckets_1-2.json \
    f_finetune/outputs/1st_run/alpaca/all_layers/ft_inf_scores/buckets_1-3.json \
    f_finetune/outputs/1st_run/alpaca/all_layers/ft_inf_scores/buckets_1-4.json \
    equiv_json/outputs/1st_run/alpaca/all_layers/ft_inf_scores/buckets_1-5.json \
    --equivalence a_data/alpaca/equi_scores/median_equi_integer.json \
    --log-log_file logs/stats_run.log \
    --output_dir e_eval/output/alpaca/tf_stats

echo "$(date) - ft_stats job finished OK"
echo "$(date) - ft_stats job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
