#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:2
#SBATCH -t 16:00:00
#SBATCH -J cmp_l8b
#SBATCH -o /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs/comparison_llama8b_%j.out
#SBATCH -e /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs/comparison_llama8b_%j.err

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
echo "Task:      Prefill vs LEMMA 1K comparison - Llama-3-8B base"
echo "=============================="

cd "${PROJECT_ROOT}"

# Data prep (fast, runs on CPU)
echo ""
echo "=== Data Preparation ==="

if [ ! -f artifacts_real/lemma_1k_original_sft_unique.json ]; then
    echo "Extracting LEMMA 1K unique..."
    ${PYTHON} scripts/data_prep/16_extract_lemma_1k_sft.py \
        --one-per-question \
        --out-file ./artifacts_real/lemma_1k_original_sft_unique.json
fi

if [ ! -f artifacts_real/lemma_sft_fix_step2_all.json ]; then
    echo "Converting prefill 245 to alpaca format..."
    ${PYTHON} scripts/data_prep/14_convert_to_lemma_format.py \
        --in-file ./artifacts_real/lemma_ds2_fix_step2_gpt_prefill.json \
        --out-file ./artifacts_real/lemma_sft_fix_step2_all.json \
        --no-require-correct
fi

echo "Data ready."

# Train + Eval + Report
echo ""
echo "=== Training + Evaluation ==="

${PYTHON} scripts/eval/29_train_and_eval_comparison.py \
    --gpus 0,1

echo ""
echo "Completed at $(date)"
