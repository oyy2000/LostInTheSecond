#!/bin/bash

#SBATCH -J 6_8_bms_deepseek
#SBATCH -p a5000ada
#SBATCH -N 1
#SBATCH -w c32
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/projects/LostInTheSecond/logs/%x/%x-%j.out
#SBATCH --error=/home/%u/projects/LostInTheSecond/logs/%x/%x-%j.err

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate fact

cd /home/youyang7/projects/LostInTheSecond

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="/home/${USER}/projects/.cache/hf_datasets/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

mkdir -p "/home/${USER}/projects/LostInTheSecond/logs/${SLURM_JOB_NAME}"
mkdir -p "${HF_DATASETS_CACHE}"

echo "Running on $(hostname)"
which python
python -V
nvidia-smi

wait_for_all_gpus_free() {
    local threshold_mib="${1:-40000}"
    while true; do
        local free_count
        free_count=$(nvidia-smi \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits \
            | awk -v t="${threshold_mib}" '$1 >= t {c++} END {print c+0}')
        if [ "${free_count}" -ge 4 ]; then
            echo "All 4 GPUs have at least ${threshold_mib} MiB free."
            nvidia-smi
            break
        fi
        echo "Waiting for free GPUs: ${free_count}/4 above ${threshold_mib} MiB at $(date)"
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader
        sleep 300
    done
}

wait_for_all_gpus_free 20000

systemd-run --user --scope -p MemoryMax=32G python -u scripts/6_9_sweep_gpt_signals.py \
    --gpus 0,1,2,3 \
    --models deepseek7b \
    --datasets gsm8k hotpotqa_open 2wikimultihopqa math500 amc2023 aime2024 olympiadbench \
    --signals prm_drop_fb_last nll_drop_fb_last \
    --n_samples 1 \
    --skip-phase 25 \
    --nd 4 8 16 
