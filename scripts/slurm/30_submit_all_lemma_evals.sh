#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/30_lemma_eval_gsm8k.sh"
ARTIFACT_ROOT=/ocean/projects/cis250050p/swang47/yang/LostInTheSecond/artifacts/comparison_llama8b

declare -A MODELS
MODELS[base]="meta-llama/Meta-Llama-3-8B"
MODELS[lora_lemma1k]="${ARTIFACT_ROOT}/lora_lemma1k/merged_model"
MODELS[lora_prefill245]="${ARTIFACT_ROOT}/lora_prefill245/merged_model"
MODELS[lora_prefill50]="${ARTIFACT_ROOT}/lora_prefill50/merged_model"
MODELS[ft_lemma1k]="${ARTIFACT_ROOT}/ft_lemma1k/best_model"
MODELS[ft_prefill245]="${ARTIFACT_ROOT}/ft_prefill245/best_model"
MODELS[ft_prefill50]="${ARTIFACT_ROOT}/ft_prefill50/best_model"

echo "Submitting 7 LEMMA GSM8K eval jobs (V100)..."
echo ""

for name in base lora_lemma1k lora_prefill245 lora_prefill50 ft_lemma1k ft_prefill245 ft_prefill50; do
    path="${MODELS[$name]}"
    JOB_ID=$(sbatch --export=ALL --parsable "${EVAL_SCRIPT}" "${name}" "${path}")
    echo "  ${name}: job ${JOB_ID}  (model: ${path})"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
