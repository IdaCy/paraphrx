#!/bin/bash
#SBATCH --job-name=gpuTest
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low
#SBATCH --gres=gpu:1
#SBATCH --time=00:03:00
#SBATCH --mem=6G
#SBATCH --chdir=/scratch_dgxl/ifc24/proj/paraphrx
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# run with the Python that lives inside the repo’s env
envs/prxenv/bin/python gpu_test.py
