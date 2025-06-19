#!/bin/bash
#SBATCH --job-name=gpuSmoke
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --gres=gpu:1
#SBATCH --time=00:02:00
#SBATCH --mem=4G
#SBATCH --chdir=/scratch_dgxl/ifc24/proj/paraphrx
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Run Python *inside* the env – no activation needed
envs/prxenv/bin/python - <<'PY'
import torch, socket, os
print("node :", socket.gethostname())
print("cwd  :", os.getcwd())
print("cuda :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu  :", torch.cuda.get_device_name(0))
PY
