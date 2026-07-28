#!/bin/bash
#SBATCH -J rebuttal_qwen_sc_agg
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/6_10/%x-%j.out
#SBATCH --error=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/6_10/%x-%j.err

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/beegfs/youyang7/.conda/envs/fact

cd /home/youyang7/projects/LostInTheSecond

export PYTHONUNBUFFERED=1

python -u rebuttal/scripts/6_10_run_qwen_math500_five_seed.py \
    --config rebuttal/configs/6_10_qwen_math500_five_seed.json \
    --aggregate-sc-only

echo "Validated five-seed SC aggregation complete"
