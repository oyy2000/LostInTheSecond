#!/usr/bin/env bash
# Phase 14: Cascade error recovery experiment on GSM8K
# Compares fixing root-cause error vs fixing cascade error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

echo "=== Phase 14_0: Find all errors and cascade relationships ==="
python "$SCRIPT_DIR/14_0_find_cascade_errors.py"

echo ""
echo "=== Phase 14_1: Generate continuations (fix_first vs fix_later) ==="
python "$SCRIPT_DIR/14_1_cascade_recovery.py" --gpus "$GPUS"

echo ""
echo "=== Phase 14_2: Analysis and figures ==="
python "$SCRIPT_DIR/14_2_cascade_analysis.py"

echo ""
echo "All done."
