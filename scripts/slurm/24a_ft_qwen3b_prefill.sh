#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:4
#SBATCH -t 08:00:00
#SBATCH -J ft_q3b_pf
#SBATCH -o /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs/ft_qwen3b_pf_%j.out
#SBATCH -e /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs/ft_qwen3b_pf_%j.err

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
echo "Task:      Full FT sweep - Qwen2.5-3B base - prefill"
echo "=============================="

cd "${PROJECT_ROOT}"

${PYTHON} scripts/full_ft/24_full_ft_sweep_base_models.py \
    --model-id Qwen/Qwen2.5-3B \
    --dataset prefill \
    --gpus 0,1,2,3

echo ""
echo "Completed at $(date)"
