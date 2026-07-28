#!/bin/bash
#SBATCH -J rebuttal_random
#SBATCH -p a5000ada
#SBATCH -N 1
#SBATCH -w c49
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/%x-%j.out
#SBATCH --error=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/%x-%j.err

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
export HF_DATASETS_CACHE="/tmp/${USER}_${SLURM_JOB_ID}_hf_datasets"
export MPLCONFIGDIR="/tmp/${USER}_${SLURM_JOB_ID}_matplotlib"

DATASET="${DATASET:-aime2025}"
RANDOM_SEED="${RANDOM_SEED:-42}"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
SOURCE_DIR="results/llama_3.2_3b_instruct_budget_multisignal/${DATASET}"
OUT_DIR="rebuttal/results/llama_3.2_3b_instruct_random/seed_${RANDOM_SEED}/${DATASET}"

mkdir -p "${HF_DATASETS_CACHE}" "${MPLCONFIGDIR}" "${OUT_DIR}"

echo "dataset=${DATASET} random_seed=${RANDOM_SEED}"
echo "source_dir=${SOURCE_DIR}"
echo "out_dir=${OUT_DIR}"
nvidia-smi

COMMON_ARGS=(
    --dataset "${DATASET}"
    --model "${MODEL}"
    --budget 32
    --gpus 0,1,2
    --signals random_uniform
    --random-seed "${RANDOM_SEED}"
    --nd 4 8 16
    --draft-checkpoint "${SOURCE_DIR}/checkpoint.jsonl"
    --out-dir "${OUT_DIR}"
)

python rebuttal/scripts/6_8_budget_controlled_random.py \
    "${COMMON_ARGS[@]}" \
    --skip-phase 5

python rebuttal/scripts/6_8_1_recount_tokens.py \
    --checkpoint "${OUT_DIR}/checkpoint.jsonl" \
    --model "${MODEL}"

python rebuttal/scripts/6_8_budget_controlled_random.py \
    "${COMMON_ARGS[@]}" \
    --skip-phase 1 4 5

python rebuttal/scripts/6_8_2_build_random_comparison.py \
    --source-summary "${SOURCE_DIR}/eval_summary.json" \
    --random-summary "${OUT_DIR}/eval_summary.json" \
    --out-dir "${OUT_DIR}" \
    --random-seed "${RANDOM_SEED}"

echo "Random baseline complete: ${OUT_DIR}"
