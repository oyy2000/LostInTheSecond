#!/bin/bash
#SBATCH --job-name=gsm8k_eval
#SBATCH --partition=a6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/beegfs/youyang7/projects/LostInSecond/logs/gsm8k_eval_%j.out
#SBATCH --error=/mnt/beegfs/youyang7/projects/LostInSecond/logs/gsm8k_eval_%j.err

# ====================================================================
# GSM8K evaluation of all LoRA adapters + base model using vLLM
# Uses 4x A6000 GPUs in parallel
#
# Submit:  sbatch scripts/slurm/gsm8k_vllm_eval.sh
# Monitor: squeue -u $USER && tail -f /mnt/beegfs/youyang7/projects/LostInSecond/logs/gsm8k_eval_*.out
# ====================================================================

set -euo pipefail

PROJECT_ROOT="/home/youyang7/projects/LostInTheSecond"
PYTHON="/mnt/beegfs/youyang7/.conda/envs/fact/bin/python"
SCRIPT="${PROJECT_ROOT}/scripts/eval/12_vllm_gsm8k_eval.py"
LOG_DIR="/mnt/beegfs/youyang7/projects/LostInSecond/logs"

mkdir -p "${LOG_DIR}"

echo "=============================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Time:      $(date)"
echo "=============================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "${PROJECT_ROOT}"

${PYTHON} "${SCRIPT}" --gpus 0,1,2,3

echo ""
echo "Job completed at $(date)"
