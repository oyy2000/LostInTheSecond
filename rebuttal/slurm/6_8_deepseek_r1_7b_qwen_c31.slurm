#!/bin/bash
#SBATCH -J rebuttal_ds7b
#SBATCH -p a6000
#SBATCH -N 1
#SBATCH -w c31
#SBATCH --cpus-per-task=64
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
export MPLCONFIGDIR="/tmp/${USER}_${SLURM_JOB_ID}_matplotlib"

DATASET="${DATASET:?Submit with --export=ALL,DATASET=math500 or hotpotqa}"
CONFIG="rebuttal/configs/6_8_deepseek_r1_7b_qwen.json"

case "${DATASET}" in
    math500|hotpotqa)
        ;;
    *)
        echo "Unsupported DATASET=${DATASET}; expected math500 or hotpotqa" >&2
        exit 2
        ;;
esac

mkdir -p "${MPLCONFIGDIR}" rebuttal/logs

echo "host=$(hostname)"
echo "dataset=${DATASET}"
echo "config=${CONFIG}"
echo "python=$(which python)"
python -V
nvidia-smi

python -u rebuttal/scripts/6_8_3_run_deepseek_rebuttal.py \
    --config "${CONFIG}" \
    --dataset "${DATASET}"

echo "Validated experiment complete: ${DATASET}"
