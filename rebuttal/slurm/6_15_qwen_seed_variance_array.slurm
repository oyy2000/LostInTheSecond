#!/bin/bash
#SBATCH -J rebuttal_qwen_5seed
#SBATCH -p a5000ada
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/%x-%A_%a.out
#SBATCH --error=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/%x-%A_%a.err

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
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-local}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
export MPLCONFIGDIR="/tmp/${USER}_${ARRAY_JOB_ID}_${ARRAY_TASK_ID}_matplotlib"

mkdir -p "${MPLCONFIGDIR}" rebuttal/logs

wait_for_gpus() {
    local threshold_mib="${GPU_FREE_THRESHOLD_MIB:-30000}"
    while true; do
        local free_count
        free_count=$(nvidia-smi \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits \
            | awk -v threshold="${threshold_mib}" \
                '$1 >= threshold {count++} END {print count+0}')
        if [ "${free_count}" -ge 3 ]; then
            return
        fi
        echo "Waiting for 3 GPUs above ${threshold_mib} MiB: ${free_count}/3"
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader
        sleep 300
    done
}

SEEDS=(42 123 456 789 1024)
DATASETS=(math500 hotpotqa_open)
SEED_INDEX=$((ARRAY_TASK_ID / 2))
DATASET_INDEX=$((ARRAY_TASK_ID % 2))
SEED="${SEEDS[$SEED_INDEX]}"
DATASET="${DATASETS[$DATASET_INDEX]}"

RESULT_ROOT="rebuttal/results/qwen2.5_3b_instruct_seed_variance"
RUN_ROOT="${RESULT_ROOT}/seed_${SEED}/${DATASET}"
POSITION_DIR="${RUN_ROOT}/position_controls"
SIGNAL_DIR="${RUN_ROOT}/nll_prm_controls"
CONFIG="rebuttal/configs/6_15_qwen_seed_variance.json"

echo "Host: $(hostname)"
echo "Dataset: ${DATASET}"
echo "Run seed: ${SEED}"

if [[ -f "${RUN_ROOT}/COMPLETE" ]]; then
    echo "Found COMPLETE marker; re-auditing before skipping generation."
    python -u rebuttal/scripts/6_10_audit_qwen_position_controls.py \
        --result-dir "${POSITION_DIR}" \
        --expected-questions 500 \
        --expected-run-seed "${SEED}"
    python -u rebuttal/scripts/6_12_audit_qwen_nll_prm_controls.py \
        --result-dir "${SIGNAL_DIR}" \
        --expected-questions 500 \
        --expected-run-seed "${SEED}"
    echo "Existing run passed both audits."
    exit 0
fi

wait_for_gpus
nvidia-smi

python -u rebuttal/scripts/6_9_qwen_position_controls.py \
    --dataset "${DATASET}" \
    --model Qwen/Qwen2.5-3B-Instruct \
    --gpus 0,1,2 \
    --n-sample 500 \
    --dataset-seed 42 \
    --random-seed "${SEED}" \
    --generation-seed "${SEED}" \
    --gpu-memory-utilization 0.32 \
    --budget 32 \
    --nd 4 8 16 \
    --strategies random_uniform \
    --out-dir "${POSITION_DIR}"

python -u rebuttal/scripts/6_10_audit_qwen_position_controls.py \
    --result-dir "${POSITION_DIR}" \
    --expected-questions 500 \
    --expected-run-seed "${SEED}"

python -u rebuttal/scripts/6_11_qwen_nll_prm_controls.py \
    --config "${CONFIG}" \
    --dataset "${DATASET}" \
    --generation-seed "${SEED}" \
    --source-dir "${POSITION_DIR}" \
    --gpus 0,1,2 \
    --out-dir "${SIGNAL_DIR}"

python -u rebuttal/scripts/6_12_audit_qwen_nll_prm_controls.py \
    --result-dir "${SIGNAL_DIR}" \
    --expected-questions 500 \
    --expected-run-seed "${SEED}"

touch "${RUN_ROOT}/COMPLETE"
