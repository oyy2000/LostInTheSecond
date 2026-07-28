#!/bin/bash
#SBATCH -J qwen_n01_p005_sc32
#SBATCH -p a5000ada
#SBATCH -N 1
#SBATCH -w c32
#SBATCH --cpus-per-task=8
#SBATCH --oversubscribe
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/%x-%j.out
#SBATCH --error=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT=/home/youyang7/projects/LostInTheSecond
ARTIFACT_ROOT=/mnt/beegfs/youyang7/projects/LostInTheSecond_rebuttal_artifacts
EXPERIMENT_NAME=qwen2.5_3b_instruct_nll_0p1_prm_0p05_sc32_recomputed
LOGICAL_RESULTS_ROOT="${PROJECT_ROOT}/rebuttal/results/${EXPERIMENT_NAME}"
PHYSICAL_RESULTS_ROOT="${ARTIFACT_ROOT}/results/${EXPERIMENT_NAME}"
LOGICAL_FIGURES_ROOT="${PROJECT_ROOT}/rebuttal/figures/${EXPERIMENT_NAME}"
PHYSICAL_FIGURES_ROOT="${ARTIFACT_ROOT}/figures/${EXPERIMENT_NAME}"
CONFIG=rebuttal/configs/6_19_qwen_nll_0p1_prm_0p05_recompute_sc32.json

if [[ -f /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh ]]; then
    source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
else
    source /home/youyang7/miniconda3/etc/profile.d/conda.sh
fi
conda activate fact

cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/beegfs/youyang7/.cache/huggingface
export HF_DATASETS_CACHE=/mnt/beegfs/youyang7/.cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="/tmp/${USER}_${SLURM_JOB_ID}_matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${MPLCONFIGDIR}" rebuttal/logs

prepare_artifact_link() {
    local logical_path="$1"
    local physical_path="$2"
    mkdir -p "$(dirname "${logical_path}")" "${physical_path}"
    if [[ ! -e "${logical_path}" && ! -L "${logical_path}" ]]; then
        ln -s "${physical_path}" "${logical_path}"
    elif [[ ! -L "${logical_path}" ]]; then
        echo "Refusing to replace non-symlink artifact path: ${logical_path}" >&2
        exit 1
    elif [[ "$(readlink -f "${logical_path}")" != \
            "$(readlink -f "${physical_path}")" ]]; then
        echo "Artifact symlink points to an unexpected target: ${logical_path}" >&2
        exit 1
    fi
}

prepare_artifact_link "${LOGICAL_RESULTS_ROOT}" "${PHYSICAL_RESULTS_ROOT}"
prepare_artifact_link "${LOGICAL_FIGURES_ROOT}" "${PHYSICAL_FIGURES_ROOT}"

IFS=' ' read -r -a DATASET_ARGS <<< \
    "${DATASETS:-gsm8k amc2023 aime2024 olympiadbench 2wikimultihopqa_open}"

# A fresh experiment must not inherit any full-SC records. Draft/PRM/NLL and
# threshold-specific suffix records are imported later by the Python runner,
# but phase 5 generates the 16 missing SC samples per question in this root.
if [[ "${ALLOW_SC32_RESUME:-0}" != "1" ]]; then
    for dataset in "${DATASET_ARGS[@]}"; do
        shared_checkpoint="${LOGICAL_RESULTS_ROOT}/seed_42/${dataset}/_shared_seed42/checkpoint.jsonl"
        if [[ -s "${shared_checkpoint}" ]]; then
            echo "Fresh SC@32 preflight failed: ${shared_checkpoint} exists." >&2
            echo "Use ALLOW_SC32_RESUME=1 only to resume an interrupted run." >&2
            exit 1
        fi
    done
fi

select_free_gpus() {
    local threshold_mib="${GPU_FREE_THRESHOLD_MIB:-28000}"
    local max_utilization="${GPU_MAX_UTILIZATION:-10}"
    local required="${GPU_COUNT:-3}"
    while true; do
        local selected
        selected=$(
            nvidia-smi \
                --query-gpu=index,memory.free,utilization.gpu \
                --format=csv,noheader,nounits \
            | awk -F, \
                -v threshold="${threshold_mib}" \
                -v max_util="${max_utilization}" \
                '$2 + 0 >= threshold && $3 + 0 <= max_util {
                    gsub(/ /, "", $1); print $1
                }' \
            | head -n "${required}" \
            | paste -sd, -
        )
        local count=0
        if [[ -n "${selected}" ]]; then
            count=$(awk -F, '{print NF}' <<< "${selected}")
        fi
        if [[ "${count}" -eq "${required}" ]]; then
            echo "${selected}"
            return
        fi
        echo "Waiting for ${required} sufficiently free GPUs: ${count}/${required}" >&2
        nvidia-smi \
            --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader >&2
        sleep 60
    done
}

echo "Host: $(hostname)"
echo "Python: $(which python)"
python -V
nvidia-smi
echo "Datasets: ${DATASET_ARGS[*]}"
echo "Thresholds: NLL=0.1, PRM=0.05"
echo "SC@32: 16 stable drafts plus 16 newly generated supplements per question"

GPU_IDS=$(select_free_gpus)
echo "Selected GPUs: ${GPU_IDS}"

python -u rebuttal/scripts/6_17_run_nll_prm_threshold_sweep.py \
    --config "${CONFIG}" \
    --gpus "${GPU_IDS}" \
    --datasets "${DATASET_ARGS[@]}"

echo "Single-threshold experiment completed and validated."
