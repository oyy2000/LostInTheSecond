#!/bin/bash
#SBATCH -J qwen_thresholds
#SBATCH -p a5000ada
#SBATCH -N 1
#SBATCH -w c32
#SBATCH --cpus-per-task=1
#SBATCH --oversubscribe
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
export HF_DATASETS_CACHE=/mnt/beegfs/youyang7/.cache/huggingface/datasets
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="/tmp/${USER}_${SLURM_JOB_ID}_matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${MPLCONFIGDIR}" rebuttal/logs

select_free_gpus() {
    local threshold_mib="${GPU_FREE_THRESHOLD_MIB:-28000}"
    local max_utilization="${GPU_MAX_UTILIZATION:-10}"
    local required="${GPU_COUNT:-2}"
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
        echo "Waiting for ${required} GPUs above ${threshold_mib} MiB and at most ${max_utilization}% utilization: ${count}/${required}" >&2
        nvidia-smi \
            --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader >&2
        sleep 300
    done
}

echo "Host: $(hostname)"
echo "Python: $(which python)"
python -V
nvidia-smi

GPU_IDS=$(select_free_gpus)
echo "Selected GPUs: ${GPU_IDS}"
IFS=' ' read -r -a DATASET_ARGS <<< \
    "${DATASETS:-math500 hotpotqa_open}"
echo "Selected datasets: ${DATASET_ARGS[*]}"

python -u rebuttal/scripts/6_17_run_nll_prm_threshold_sweep.py \
    --config rebuttal/configs/6_17_qwen_nll_prm_threshold_sweep.json \
    --gpus "${GPU_IDS}" \
    --datasets "${DATASET_ARGS[@]}"

echo "Threshold sweep completed and validated."
