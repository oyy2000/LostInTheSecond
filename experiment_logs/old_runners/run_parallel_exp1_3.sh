#!/bin/bash
set -uo pipefail

# Parallel launcher: Exp 1-2 + Exp 3
# GPU isolation: scripts parse --gpus and set CUDA_VISIBLE_DEVICES before torch import.
# Phase A: Exp 3 (fast) on GPUs 1-3 + 7, Exp 1-2 for 0.5B on GPU 0
# Phase B: Exp 1-2 for 1.5B, 3B, 7B sequentially on GPU 7

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/common/home/sl2148/anaconda3/envs/fact_yang/bin/python"
LOGDIR="$PROJECT_ROOT/runs/parallel_logs"
mkdir -p "$LOGDIR"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES

echo "============================================================"
echo "  Parallel Exp 1-2 + Exp 3"
echo "  $(date)"
echo "============================================================"
nvidia-smi --query-gpu=index,memory.free --format=csv
echo ""

declare -A PIDS

# ── Phase A: Exp 3 (all 4 models) + Exp 1-2 (0.5B) ──
echo "=== Phase A: Exp 3 (all models) + Exp 1-2 (0.5B) ==="
echo ""

$PYTHON "$SCRIPT_DIR/24_hidden_state_mode_analysis.py" --models 0.5B --gpus 1 > "$LOGDIR/exp3_0.5B.log" 2>&1 &
PIDS[exp3_0.5B]=$!
echo "[LAUNCH] exp3_0.5B on GPU 1 (PID ${PIDS[exp3_0.5B]})"

$PYTHON "$SCRIPT_DIR/24_hidden_state_mode_analysis.py" --models 1.5B --gpus 2 > "$LOGDIR/exp3_1.5B.log" 2>&1 &
PIDS[exp3_1.5B]=$!
echo "[LAUNCH] exp3_1.5B on GPU 2 (PID ${PIDS[exp3_1.5B]})"

$PYTHON "$SCRIPT_DIR/24_hidden_state_mode_analysis.py" --models 3B --gpus 3 > "$LOGDIR/exp3_3B.log" 2>&1 &
PIDS[exp3_3B]=$!
echo "[LAUNCH] exp3_3B on GPU 3 (PID ${PIDS[exp3_3B]})"

$PYTHON "$SCRIPT_DIR/24_hidden_state_mode_analysis.py" --models 7B --gpus 7 > "$LOGDIR/exp3_7B.log" 2>&1 &
PIDS[exp3_7B]=$!
echo "[LAUNCH] exp3_7B on GPU 7 (PID ${PIDS[exp3_7B]})"

$PYTHON "$SCRIPT_DIR/23_good_prefix_injection.py" --models 0.5B --gpus 0 > "$LOGDIR/exp1_0.5B.log" 2>&1 &
PIDS[exp1_0.5B]=$!
echo "[LAUNCH] exp1_0.5B on GPU 0 (PID ${PIDS[exp1_0.5B]})"

echo ""
echo "Phase A: 5 processes launched. Waiting for Exp 3..."
echo ""

FAILED=0
for name in exp3_0.5B exp3_1.5B exp3_3B exp3_7B; do
    pid=${PIDS[$name]}
    wait "$pid"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] $name (PID $pid) exited with code $rc"
        FAILED=$((FAILED + 1))
    else
        echo "[OK]   $name (PID $pid) completed"
    fi
done

echo ""
echo "=== Exp 3 done ($((4 - FAILED))/4 succeeded). $(date) ==="
echo ""

echo "Running Exp 3 plot aggregation..."
$PYTHON "$SCRIPT_DIR/24_hidden_state_mode_analysis.py" --only-plot --models "0.5B,1.5B,3B,7B" 2>&1 | tee "$LOGDIR/exp3_plot_aggregation.log"

# ── Phase B: Exp 1-2 for 1.5B, 3B, 7B on GPU 7 ──
echo ""
echo "=== Phase B: Exp 1-2 for 1.5B, 3B, 7B on GPU 7 ==="
echo "(0.5B still running on GPU 0)"
echo ""

for tag in 1.5B 3B 7B; do
    echo "[RUN] Exp 1-2 for $tag on GPU 7 ($(date))"
    $PYTHON "$SCRIPT_DIR/23_good_prefix_injection.py" --models "$tag" --gpus 7 > "$LOGDIR/exp1_${tag}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] exp1_$tag exited with code $rc"
    else
        echo "[OK]   exp1_$tag completed"
    fi
done

echo ""
echo "Waiting for exp1_0.5B to finish..."
wait ${PIDS[exp1_0.5B]} 2>/dev/null
rc=$?
if [ $rc -ne 0 ]; then
    echo "[FAIL] exp1_0.5B exited with code $rc"
else
    echo "[OK]   exp1_0.5B completed"
fi

echo ""
echo "Running Exp 1-2 plot aggregation..."
$PYTHON "$SCRIPT_DIR/23_good_prefix_injection.py" --only-plot --models "0.5B,1.5B,3B,7B" 2>&1 | tee "$LOGDIR/exp1_plot_aggregation.log"

echo ""
echo "============================================================"
echo "  ALL DONE at $(date)"
echo "============================================================"
echo "Results:"
echo "  Exp 1-2: $PROJECT_ROOT/runs/good_prefix_exp/"
echo "  Exp 3:   $PROJECT_ROOT/runs/hidden_state_mode/"
echo "  Logs:    $LOGDIR/"
