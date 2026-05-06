#!/usr/bin/env bash
# Run Phase 2 method sweep: Llama 3.2 3B x all 8 datasets
#
# n_drafts = 1,2,4,8,16,32
# K = 2,3,4  (per-draft budget: 1 greedy + K-1 suffixes)
#
# Checkpoint/resume: safe to kill and re-run; completed tasks are skipped.
#
# Usage:
#   bash scripts/15_2_run_llama3b_sweep.sh          # default 8 GPUs
#   GPUS=0,1,2,3 bash scripts/15_2_run_llama3b_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

MODEL="meta-llama/Llama-3.2-3B-Instruct"
DRAFTS="2,4,8"
KS="3,4"
ALPHAS="0.5"

DATASETS=(
    hotpotqa
    math500
    gsm8k
    aime2024
    amc2023
    olympiadbench
    humaneval
    csqa
)

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "  ${MODEL}  x  ${DS}"
    echo "  n_drafts=${DRAFTS}  K=${KS}  alphas=${ALPHAS}"
    echo "======================================================================"
    python "$SCRIPT_DIR/15_1_method_sweep.py" \
        --model-id "$MODEL" \
        --dataset "$DS" \
        --gpus "$GPUS" \
        --n-drafts-list "$DRAFTS" \
        --Ks "$KS" \
        --alphas "$ALPHAS"
done

echo ""
echo "All datasets done."
