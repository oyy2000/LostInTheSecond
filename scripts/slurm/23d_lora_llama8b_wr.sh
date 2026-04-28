#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:4
#SBATCH -t 12:00:00
#SBATCH -J lora_l8b_wr
#SBATCH -o /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs/lora_llama8b_wr_%j.out
#SBATCH -e /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs/lora_llama8b_wr_%j.err

set -euo pipefail

PROJECT_ROOT="/jet/home/swang47/yang/projects/LostInTheSecond"
PYTHON="/ocean/projects/cis250050p/swang47/miniconda3/envs/sft_yang/bin/python"

export CUDA_HOME=/opt/packages/cuda/v12.1.1
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs

echo "=============================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Time:      $(date)"
echo "Task:      LoRA sweep - Llama-3-8B base - wait+recompute"
echo "=============================="

cd "${PROJECT_ROOT}"

${PYTHON} scripts/lora/23_lora_sweep_base_models.py \
    --model-id meta-llama/Meta-Llama-3-8B \
    --dataset wait_recompute \
    --gpus 0,1,2,3

echo ""
echo "Completed at $(date)"
