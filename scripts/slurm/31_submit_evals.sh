#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/30_lemma_eval_gsm8k.sh"
ARTIFACT_ROOT=/ocean/projects/cis250050p/swang47/yang/LostInTheSecond/artifacts/full_prefill_llama8b

echo "Submitting 5 GSM8K eval jobs (V100, 1 GPU each)..."
echo ""

declare -A MODELS
MODELS[base]="meta-llama/Meta-Llama-3-8B"
MODELS[lora_fix498]="${ARTIFACT_ROOT}/lora_fix498/merged_model"
MODELS[lora_wr2224]="${ARTIFACT_ROOT}/lora_wr2224/merged_model"
MODELS[ft_fix498]="${ARTIFACT_ROOT}/ft_fix498/best_model"
MODELS[ft_wr2224]="${ARTIFACT_ROOT}/ft_wr2224/best_model"

for name in base lora_fix498 lora_wr2224 ft_fix498 ft_wr2224; do
    path="${MODELS[$name]}"
    JOB_ID=$(sbatch --export=ALL --parsable "${EVAL_SCRIPT}" "${name}" "${path}")
    echo "  ${name}: job ${JOB_ID}  (model: ${path})"
done

echo ""
echo "All eval jobs submitted. Monitor with: squeue -u \$USER"
