#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
N_SAMPLES="${N_SAMPLES:-8}"
N_CONTINUATIONS="${N_CONTINUATIONS:-32}"
LIMIT="${LIMIT:-0}"

echo "=== Phase 13: MBPP validation ==="
echo "GPUs: $GPUS | N_SAMPLES: $N_SAMPLES | N_CONT: $N_CONTINUATIONS"

echo "--- 13.0: Sample multi-CoT ---"
python "$SCRIPT_DIR/13_0_mbpp_sample_multi_cot.py" \
    --n-samples "$N_SAMPLES" --gpus "$GPUS" --limit "$LIMIT"

echo "--- 13.1: Find first error + bucket ---"
python "$SCRIPT_DIR/13_1_find_first_error_and_bucket.py" \
    --model gpt-5.1 --max-workers 8

echo "--- 13.2: Bad-prefix natural recovery ---"
python "$SCRIPT_DIR/13_2_bad_prefix_natural_recovery.py" \
    --n-continuations "$N_CONTINUATIONS" --gpus "$GPUS"

echo "--- 13.3: Minimal-repair continuation ---"
python "$SCRIPT_DIR/13_3_minimal_repair_continuation.py" \
    --n-continuations "$N_CONTINUATIONS" --gpus "$GPUS"

echo "--- 13.4: Compare bad vs repair ---"
python "$SCRIPT_DIR/13_4_compare_bad_vs_repair.py"
python "$SCRIPT_DIR/13_4_compare_bad_vs_repair.py" --per-question

echo "--- 13.5: Fine-grained relpos ---"
python "$SCRIPT_DIR/13_5_fine_grained_relpos.py"

echo "=== Phase 13 complete ==="
echo "Results: results/mbpp_3b_multi_sample/"
echo "Figures: figures/mbpp_bad_prefix_recovery/"
