#!/usr/bin/env bash
# ============================================================
# Phase 12: HumanEval validation pipeline
#
# Validates the "relative error position -> recovery" motivation
# on HumanEval (function-level code generation).
#
# Prerequisites:
#   - vLLM, datasets, openai, scipy, matplotlib, tqdm
#   - OPENAI_API_KEY in .env or environment
#   - 8 GPUs (adjust --gpus as needed)
#
# Usage:
#   bash scripts/12_run_all.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
N_SAMPLES="${N_SAMPLES:-8}"
N_CONTINUATIONS="${N_CONTINUATIONS:-32}"
LIMIT="${LIMIT:-0}"

echo "=== Phase 12: HumanEval validation ==="
echo "GPUs: $GPUS | N_SAMPLES: $N_SAMPLES | N_CONT: $N_CONTINUATIONS | LIMIT: $LIMIT"
echo ""

# Step 0: Sample multiple code solutions per problem
echo "--- Step 12.0: Sample multi-CoT code solutions ---"
python "$SCRIPT_DIR/12_0_humaneval_sample_multi_cot.py" \
    --n-samples "$N_SAMPLES" \
    --gpus "$GPUS" \
    --limit "$LIMIT"
echo ""

# Step 1: Find first-error block via GPT + bucket
echo "--- Step 12.1: Find first error block + bucket ---"
python "$SCRIPT_DIR/12_1_find_first_error_and_bucket.py" \
    --model gpt-5.1 \
    --max-workers 8
echo ""

# Step 2: Bad-prefix natural recovery
echo "--- Step 12.2: Bad-prefix natural recovery ---"
python "$SCRIPT_DIR/12_2_bad_prefix_natural_recovery.py" \
    --n-continuations "$N_CONTINUATIONS" \
    --gpus "$GPUS"
echo ""

# Step 3: Minimal-repair continuation
echo "--- Step 12.3: Minimal-repair continuation ---"
python "$SCRIPT_DIR/12_3_minimal_repair_continuation.py" \
    --n-continuations "$N_CONTINUATIONS" \
    --gpus "$GPUS"
echo ""

# Step 4: Compare bad-prefix vs repair (figures + stats)
echo "--- Step 12.4: Compare bad vs repair ---"
python "$SCRIPT_DIR/12_4_compare_bad_vs_repair.py"
python "$SCRIPT_DIR/12_4_compare_bad_vs_repair.py" --per-question
echo ""

# Step 5: Fine-grained relative position analysis
echo "--- Step 12.5: Fine-grained relpos analysis ---"
python "$SCRIPT_DIR/12_5_fine_grained_relpos.py"
echo ""

echo "=== Phase 12 complete ==="
echo "Results: results/humaneval_3b_multi_sample/"
echo "Figures: figures/humaneval_bad_prefix_recovery/"
