#!/bin/bash
#SBATCH -J rebuttal_qwen_5seed_summary
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/%x-%j.out
#SBATCH --error=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/%x-%j.err

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/beegfs/youyang7/.conda/envs/fact

cd /home/youyang7/projects/LostInTheSecond

export MPLCONFIGDIR="/tmp/${USER}_${SLURM_JOB_ID}_matplotlib"
mkdir -p "${MPLCONFIGDIR}" rebuttal/logs

python -u rebuttal/scripts/6_15_summarize_qwen_seed_variance.py
