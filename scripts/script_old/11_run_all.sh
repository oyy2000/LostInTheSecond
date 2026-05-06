#!/usr/bin/env bash
# ============================================================
# Phase 11: CodeContests (AlphaCode) validation pipeline
#
# Validates the "relative error position -> recovery" motivation
# on competitive programming (code generation) tasks.
#
# Prerequisites:
#   - vLLM, datasets, openai, scipy, matplotlib, tqdm
#   - OPENAI_API_KEY in .env or environment
#   - 8 GPUs (adjust --gpus as needed)
#
# Usage:
#   bash scripts/11_run_all.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
N_SAMPLES="${N_SAMPLES:-8}"
N_CONTINUATIONS="${N_CONTINUATIONS:-32}"
LIMIT="${LIMIT:-0}"  # 0 = all problems

echo "=== Phase 11: CodeContests validation ==="
echo "GPUs: $GPUS | N_SAMPLES: $N_SAMPLES | N_CONT: $N_CONTINUATIONS | LIMIT: $LIMIT"
echo ""

# Step 0: Sample multiple code solutions per problem
echo "--- Step 11.0: Sample multi-CoT code solutions ---"
python "$SCRIPT_DIR/11_0_codecontests_sample_multi_cot.py" \
    --n-samples "$N_SAMPLES" \
    --gpus "$GPUS" \
    --limit "$LIMIT"
echo ""

# Step 1: Find first-error block via GPT + bucket
echo "--- Step 11.1: Find first error block + bucket ---"
python "$SCRIPT_DIR/11_1_find_first_error_and_bucket.py" \
    --model gpt-5.1 \
    --max-workers 8
echo ""

# Step 2: Bad-prefix natural recovery
echo "--- Step 11.2: Bad-prefix natural recovery ---"
python "$SCRIPT_DIR/11_2_bad_prefix_natural_recovery.py" \
    --n-continuations "$N_CONTINUATIONS" \
    --gpus "$GPUS"
echo ""

# Step 3: Minimal-repair continuation
echo "--- Step 11.3: Minimal-repair continuation ---"
python "$SCRIPT_DIR/11_3_minimal_repair_continuation.py" \
    --n-continuations "$N_CONTINUATIONS" \
    --gpus "$GPUS"
echo ""

# Step 4: Compare bad-prefix vs repair (figures + stats)
echo "--- Step 11.4: Compare bad vs repair ---"
python "$SCRIPT_DIR/11_4_compare_bad_vs_repair.py"
python "$SCRIPT_DIR/11_4_compare_bad_vs_repair.py" --per-question
echo ""

# Step 5: Fine-grained relative position analysis
echo "--- Step 11.5: Fine-grained relpos analysis ---"
python "$SCRIPT_DIR/11_5_fine_grained_relpos.py"
echo ""

echo "=== Phase 11 complete ==="
echo "Results: results/codecontests_3b_multi_sample/"
echo "Figures: figures/codecontests_bad_prefix_recovery/"
