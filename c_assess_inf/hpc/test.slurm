#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=1G
#SBATCH --time=00:00:10
#SBATCH -o /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_test.out
#SBATCH -e /scratch_dgxl/ifc24/proj/paraphrx/logs/%x_%j_test.err

echo "$(date) – test job started on $(hostname)"

echo "$(date) – test job finished OK" >> "$SLURM_SUBMIT_DIR/times.log"

echo "$(date) – test job finished OK"
