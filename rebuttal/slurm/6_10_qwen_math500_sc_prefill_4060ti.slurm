#!/bin/bash
#SBATCH -J rebuttal_qwen_sc
#SBATCH -p rtx4060ti16g
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --array=0-4%5
#SBATCH --time=06:00:00
#SBATCH --output=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/6_10/%x-%A_%a.out
#SBATCH --error=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/6_10/%x-%A_%a.err

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/beegfs/youyang7/.conda/envs/fact

cd /home/youyang7/projects/LostInTheSecond

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/beegfs/youyang7/.cache/huggingface
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MPLCONFIGDIR="/tmp/${USER}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_matplotlib"

SEEDS=(42 123 456 789 1024)
RUN_SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
CONFIG="rebuttal/configs/6_10_qwen_math500_five_seed.json"

mkdir -p "${MPLCONFIGDIR}" rebuttal/logs/6_10

echo "host=$(hostname)"
echo "array_job=${SLURM_ARRAY_JOB_ID}"
echo "array_task=${SLURM_ARRAY_TASK_ID}"
echo "run_seed=${RUN_SEED}"
echo "mode=generation-only"
echo "python=$(which python)"
python -V
nvidia-smi

python -u rebuttal/scripts/6_10_run_qwen_math500_five_seed.py \
    --config "${CONFIG}" \
    --gpus 0 \
    --seeds "${RUN_SEED}" \
    --generation-only

echo "Validated corrected SC generation complete: ${RUN_SEED}"
