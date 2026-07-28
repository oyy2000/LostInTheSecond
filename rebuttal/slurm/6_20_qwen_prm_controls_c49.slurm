#!/bin/bash
#SBATCH -J qwen_prm_controls
#SBATCH -p a5000ada
#SBATCH -N 1
#SBATCH -w c49
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/%x-%j.out
#SBATCH --error=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/%x-%j.err

set -euo pipefail

if [[ -f /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh ]]; then
    source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
else
    source /home/youyang7/miniconda3/etc/profile.d/conda.sh
fi
conda activate fact

cd /home/youyang7/projects/LostInTheSecond

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/beegfs/youyang7/.cache/huggingface
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="/tmp/${USER}_${SLURM_JOB_ID}_matplotlib"
mkdir -p "${MPLCONFIGDIR}" rebuttal/logs

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
        echo "Waiting for ${required} free GPUs: ${count}/${required}" >&2
        nvidia-smi \
            --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader >&2
        sleep 300
    done
}

DATASET="${DATASET:-math500}"
case "${DATASET}" in
    math500|hotpotqa_open) ;;
    *)
        echo "Unsupported DATASET=${DATASET}" >&2
        exit 2
        ;;
esac

echo "Host: $(hostname)"
echo "Dataset: ${DATASET}"
echo "Python: $(which python)"
python -V
nvidia-smi

GPU_IDS=$(select_free_gpus)
echo "Selected GPUs: ${GPU_IDS}"

python -u rebuttal/scripts/6_20_run_prm_control_experiments.py \
    --config rebuttal/configs/6_20_qwen_prm_control_experiments.json \
    --datasets "${DATASET}" \
    --gpus "${GPU_IDS}"

echo "PRM control experiment completed and validated: ${DATASET}"
