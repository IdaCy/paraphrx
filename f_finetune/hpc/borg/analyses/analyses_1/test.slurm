#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --partition=root
#SBATCH --qos=intermediate
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch_root/ifc24/paraphrx/logs/%j_%x.out
#SBATCH --error=/scratch_root/ifc24/paraphrx/logs/%j_%x.err

echo "$(date) - finished test!"
