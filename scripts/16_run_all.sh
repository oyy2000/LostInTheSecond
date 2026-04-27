#!/usr/bin/env bash
# Run the full state-aware late rollback pipeline.
#
# Steps:
#   1. Entropy-guided LATER (16_0): logprob scoring + dynamic rollback + suffix
#   2. Probe training (16_1): extract hidden states, train linear probe
#   3. Probe-guided LATER (16_2): probe scoring + dynamic rollback + suffix
#   4. Composite-guided LATER (16_3): combined scoring + grid search
#   5. Analysis + figures (16_4): compare all methods vs oracle + fixed-alpha
#
# Usage:
#   bash scripts/16_run_all.sh
#   GPUS=0,1,2,3 bash scripts/16_run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PYTHON="/common/users/sl2148/anaconda3/envs/tokenskip_yang/bin/python"

MODEL="meta-llama/Llama-3.2-3B-Instruct"
DATASET="gsm8k"

echo "============================================================"
echo "State-Aware Late Rollback Pipeline"
echo "  Model:   $MODEL"
echo "  Dataset: $DATASET"
echo "  GPUs:    $GPUS"
echo "  Python:  $PYTHON"
echo "============================================================"

# Step 1: Entropy-guided LATER
echo ""
echo "[Step 1/5] Entropy-guided LATER (16_0)"
$PYTHON "$SCRIPT_DIR/16_0_entropy_guided_later.py" \
    --model-id "$MODEL" \
    --dataset "$DATASET" \
    --gpus "$GPUS" \
    --n-drafts 4 \
    --K 4 \
    --betas "0.4,0.5,0.6" \
    --methods "argmax,max_drop"

# Step 2: Probe training
echo ""
echo "[Step 2/5] Probe training (16_1)"
# Use first available GPU for probe training (single-GPU HF forward pass)
FIRST_GPU=$(echo "$GPUS" | cut -d',' -f1)
$PYTHON "$SCRIPT_DIR/16_1_train_rollback_probe.py" \
    --model-id "$MODEL" \
    --gpus "$FIRST_GPU" \
    --layers "8,16,24,28" \
    --max-samples 1000

# Step 3: Probe-guided LATER
echo ""
echo "[Step 3/5] Probe-guided LATER (16_2)"
$PYTHON "$SCRIPT_DIR/16_2_probe_guided_later.py" \
    --model-id "$MODEL" \
    --dataset "$DATASET" \
    --gpus "$GPUS" \
    --scoring-gpu "$FIRST_GPU" \
    --n-drafts 4 \
    --K 4 \
    --betas "0.4,0.5,0.6" \
    --methods "argmax,max_drop"

# Step 4: Composite-guided LATER
echo ""
echo "[Step 4/5] Composite-guided LATER (16_3)"
$PYTHON "$SCRIPT_DIR/16_3_composite_guided_later.py" \
    --model-id "$MODEL" \
    --dataset "$DATASET" \
    --gpus "$GPUS" \
    --n-drafts 4 \
    --K 4 \
    --beta 0.5

# Step 5: Analysis + figures
echo ""
echo "[Step 5/5] Analysis + comparison figures (16_4)"
$PYTHON "$SCRIPT_DIR/16_4_rollback_point_analysis.py" \
    --model-id "$MODEL" \
    --dataset "$DATASET" \
    --n-drafts 4 \
    --beta 0.5

echo ""
echo "============================================================"
echo "Pipeline complete. Results in:"
echo "  results/${DATASET}_*_entropy_later/"
echo "  results/${DATASET}_*_rollback_probe/"
echo "  results/${DATASET}_*_probe_later/"
echo "  results/${DATASET}_*_composite_later/"
echo "  figures/state_aware_rollback/"
echo "============================================================"
