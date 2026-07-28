#!/bin/bash

#SBATCH -J 6_8_bms_math500
#SBATCH -p a5000ada
#SBATCH -N 1
#SBATCH -w c32
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/projects/logs/%x/%x-%j.out
#SBATCH --error=/home/%u/projects/logs/%x/%x-%j.err

set -euo pipefail

if [ -f /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh ]; then
    source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
else
    source /home/youyang7/miniconda3/etc/profile.d/conda.sh
fi
conda activate fact

cd /home/youyang7/projects/LostInTheSecond

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="/home/${USER}/projects/.cache/hf_datasets/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

mkdir -p "/home/${USER}/projects/logs/${SLURM_JOB_NAME}"
mkdir -p "${HF_DATASETS_CACHE}"

echo "Running on $(hostname)"
which python
python -V
nvidia-smi

wait_for_gpus_free() {
    local threshold_mib="${1:-30000}"
    local required="${2:-3}"
    while true; do
        local free_count
        free_count=$(nvidia-smi \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits \
            | awk -v t="${threshold_mib}" '$1 >= t {c++} END {print c+0}')
        if [ "${free_count}" -ge "${required}" ]; then
            echo "${free_count} GPUs have at least ${threshold_mib} MiB free."
            nvidia-smi
            break
        fi
        echo "Waiting for GPUs: ${free_count}/${required} above ${threshold_mib} MiB at $(date)"
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader
        sleep 300
    done
}

wait_for_gpus_free 30000 3

python scripts/6_8_budget_controlled_multisignal.py \
    --dataset math500 \
    --budget 32 \
    --gpus 0,1,2 \
    --signals gpt prm_drop nll_drop \
    --prm-drop-threshold 0.05 \
    --nll-drop-threshold 0.1 \
    --gpt-cache results/math500_3b_multi_sample/first_error/gpt_first_error_cache.jsonl \
    --tag budget32_c32_math500
