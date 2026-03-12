#!/bin/bash
#SBATCH --job-name=lora_sweep
#SBATCH --partition=a6000
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=/mnt/beegfs/youyang7/projects/LostInSecond/logs/lora_sweep_%j.out
#SBATCH --error=/mnt/beegfs/youyang7/projects/LostInSecond/logs/lora_sweep_%j.err

# ====================================================================
# LoRA hyperparameter sweep Phase 3 (full tuning)
# Re-trains failed experiments + new configs, then evaluates on GSM8K
#
# Submit:  sbatch scripts/slurm/lora_sweep_v3.sh
# ====================================================================

set -euo pipefail

PROJECT_ROOT="/home/youyang7/projects/LostInTheSecond"
PYTHON="/mnt/beegfs/youyang7/.conda/envs/fact/bin/python"
LOG_DIR="/mnt/beegfs/youyang7/projects/LostInSecond/logs"

mkdir -p "${LOG_DIR}"

echo "=============================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Time:      $(date)"
echo "=============================="

cd "${PROJECT_ROOT}"

# Phase 1: Train all configs in parallel on 4 GPUs
${PYTHON} scripts/lora/11_sweep_v2_and_eval.py --phase train-only --gpus 0,1,2,3

# Phase 2: Evaluate all adapters (vLLM, parallel)
${PYTHON} scripts/eval/12_vllm_gsm8k_eval.py --gpus 0,1,2,3

echo ""
echo "Sweep completed at $(date)"
