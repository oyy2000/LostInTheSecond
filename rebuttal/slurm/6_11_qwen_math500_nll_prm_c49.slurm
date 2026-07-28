#!/bin/bash
#SBATCH -J rebuttal_qwen_math_signals
#SBATCH -p a5000ada
#SBATCH -N 1
#SBATCH -w c49
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/%x-%j.out
#SBATCH --error=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/%x-%j.err

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="/tmp/${USER}_${SLURM_JOB_ID}_matplotlib"

mkdir -p "${MPLCONFIGDIR}" rebuttal/logs

wait_for_gpus() {
    local threshold_mib=30000
    while true; do
        local free_count
        free_count=$(nvidia-smi \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits \
            | awk -v threshold="${threshold_mib}" \
                '$1 >= threshold {count++} END {print count+0}')
        if [ "${free_count}" -ge 3 ]; then
            break
        fi
        echo "Waiting for 3 GPUs above ${threshold_mib} MiB: ${free_count}/3"
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader
        sleep 300
    done
}

echo "Host: $(hostname)"
wait_for_gpus
nvidia-smi

RESULT_DIR="rebuttal/results/qwen2.5_3b_instruct_nll_prm_controls/seed_42/math500"

python -u rebuttal/scripts/6_11_qwen_nll_prm_controls.py \
    --config rebuttal/configs/6_11_qwen_nll_prm_controls.json \
    --dataset math500 \
    --gpus 0,1,2 \
    --out-dir "${RESULT_DIR}"

python -u rebuttal/scripts/6_12_audit_qwen_nll_prm_controls.py \
    --result-dir "${RESULT_DIR}" \
    --expected-questions 500
