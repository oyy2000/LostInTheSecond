#!/bin/bash
set -uo pipefail

# Parallel launcher: 8 models across 8 GPUs for Exp 0
# 4 Instruct (GPUs 0-3) + 4 Base (GPUs 4-7)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/common/home/sl2148/anaconda3/envs/fact_yang/bin/python"
LOGDIR="$PROJECT_ROOT/runs/parallel_logs"
mkdir -p "$LOGDIR"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "  Parallel Exp 0: 8 models on 8 GPUs"
echo "  $(date)"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.free --format=csv
echo ""

SCRIPT="$SCRIPT_DIR/22_multi_scale_baseline.py"

declare -A PIDS

launch() {
    local tag="$1"
    local gpu="$2"
    local logfile="$LOGDIR/${tag}.log"
    echo "[LAUNCH] $tag on GPU $gpu -> $logfile"
    $PYTHON "$SCRIPT" --models "$tag" --gpus "$gpu" > "$logfile" 2>&1 &
    PIDS["$tag"]=$!
}

# Instruct models on GPUs 0-3
launch "0.5B"  0
launch "1.5B"  1
launch "3B"    2
launch "7B"    3

# Base models on GPUs 4-7
launch "0.5B-base" 4
launch "1.5B-base" 5
launch "3B-base"   6
launch "7B-base"   7

echo ""
echo "All 8 processes launched. PIDs:"
for tag in "${!PIDS[@]}"; do
    echo "  $tag: ${PIDS[$tag]}"
done
echo ""
echo "Logs: $LOGDIR/"
echo "Monitor: tail -f $LOGDIR/*.log"
echo ""

FAILED=0
for tag in "${!PIDS[@]}"; do
    pid=${PIDS[$tag]}
    wait "$pid"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] $tag (PID $pid) exited with code $rc"
        FAILED=$((FAILED + 1))
    else
        echo "[OK]   $tag (PID $pid) completed successfully"
    fi
done

echo ""
echo "============================================================"
echo "  Phase 1 complete: $((8 - FAILED))/8 succeeded"
echo "  $(date)"
echo "============================================================"

if [ $FAILED -gt 0 ]; then
    echo "[WARN] $FAILED processes failed. Check logs in $LOGDIR/"
fi

# Final aggregation: cross-scale plots for instruct models
echo ""
echo "Running final aggregation (instruct models)..."
$PYTHON "$SCRIPT" --only-plot --models "0.5B,1.5B,3B,7B" 2>&1 | tee "$LOGDIR/aggregation_instruct.log"

# Aggregation for base models (separate output dir)
echo ""
echo "Running final aggregation (base models)..."
$PYTHON "$SCRIPT" --only-plot --models "0.5B-base,1.5B-base,3B-base,7B-base" 2>&1 | tee "$LOGDIR/aggregation_base.log"

# Run CPU-only downstream experiments immediately
echo ""
echo "Running Exp 4: Conditioned PRM Analysis (CPU)..."
$PYTHON "$SCRIPT_DIR/25_conditioned_prm_analysis.py" --models "0.5B,1.5B,3B,7B" 2>&1 | tee "$LOGDIR/exp4_conditioned.log"

echo ""
echo "Running Exp 5: Scaling Law Analysis (CPU)..."
$PYTHON "$SCRIPT_DIR/26_scaling_law_mode_stability.py" 2>&1 | tee "$LOGDIR/exp5_scaling.log"

echo ""
echo "============================================================"
echo "  ALL DONE at $(date)"
echo "============================================================"
echo "Results:"
echo "  Baselines:   $PROJECT_ROOT/runs/multi_scale_baselines/"
echo "  PRM Scores:  $PROJECT_ROOT/runs/multi_scale_prm/"
echo "  Conditioned: $PROJECT_ROOT/runs/conditioned_prm/"
echo "  Scaling Law: $PROJECT_ROOT/runs/scaling_law_analysis/"
echo "  Logs:        $LOGDIR/"
