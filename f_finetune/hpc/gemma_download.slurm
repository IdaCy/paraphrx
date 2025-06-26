#!/bin/bash -l
#SBATCH --job-name=gemma_dl
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH -o /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_gemma_dl.out
#SBATCH -e /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_gemma_dl.err
set -euo pipefail

echo "$(date) – gemma_dl job started on $(hostname)"
echo "$(date) – gemma_dl job started OK" >> "$SLURM_SUBMIT_DIR/times.log"

# Workspace & caches
cd /scratch_dgxl/ifc24/proj/paraphrx

mkdir -p logs tmp hf_cache pip_cache xdg_cache rust_cache
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

# Rust tools
export CARGO_HOME=$PWD/rust_cache/cargo
export RUSTUP_HOME=$PWD/rust_cache/rustup
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"

if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
    echo "Installing minimal Rust toolchain locally …"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --default-toolchain stable --profile minimal
fi
export PATH="$CARGO_HOME/bin:$PATH"

# Hugging Face – faster transfer & token
export HF_HUB_ENABLE_HF_TRANSFER=1
source /scratch_dgxl/ifc24/.hf_token

# Build once (release) and run the downloader through srun
echo "Building gemma_download …"
cargo build --manifest-path f_finetune/Cargo.toml \
            --bin gemma_download --release

echo "Starting Gemma download …"
srun cargo gemma_download \
        --token "$HF_TOKEN" \
        --dst f_finetune/model

echo "$(date) – gemma_dl job finished OK"
echo "$(date) – gemma_dl job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"
