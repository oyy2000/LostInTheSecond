#!/bin/bash
set -uo pipefail

# ============================================================
#  Parallel launcher: Exp 7 (Representation Injection)
#  + Re-run 7B Exp 1-2 (was killed in prior run)
#
#  GPU 0,1: 7B Exp 1-2 re-run (2-GPU split)
#  GPU 2:   Exp 7 — 0.5B
#  GPU 3:   Exp 7 — 1.5B
#  GPU 4:   Exp 7 — 3B
#  GPU 5,6: Exp 7 — 7B (2-GPU split)
#  GPU 7:   free (or idle)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/common/home/sl2148/anaconda3/envs/fact_yang/bin/python"
LOGDIR="$PROJECT_ROOT/runs/parallel_logs"
mkdir -p "$LOGDIR"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES

# Kill stuck processes from prior runs
echo "Cleaning up stuck processes..."
pkill -f "run_parallel_missing.sh" 2>/dev/null || true
pkill -f "run_parallel_exp1_3.sh" 2>/dev/null || true
sleep 5

echo "============================================================"
echo "  Parallel Exp 7 + 7B Exp 1-2 Re-run"
echo "  Started: $(date)"
echo "============================================================"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
echo ""

declare -A PIDS

# ── 7B Exp 1-2 re-run on GPUs 0,1 ──
echo "[LAUNCH] 7B Exp 1-2 on GPUs 0,1"
$PYTHON "$SCRIPT_DIR/23_good_prefix_injection.py" \
    --models 7B \
    --gpus 0,1 \
    > "$LOGDIR/exp1_7B_2gpu_rerun.log" 2>&1 &
PIDS[exp1_7B]=$!
echo "  PID: ${PIDS[exp1_7B]}"

# ── Exp 7: Representation Injection ──
echo "[LAUNCH] Exp 7 — 0.5B on GPU 2"
$PYTHON "$SCRIPT_DIR/27_representation_injection.py" \
    --models 0.5B \
    --gpus 2 \
    > "$LOGDIR/exp7_0.5B.log" 2>&1 &
PIDS[exp7_0.5B]=$!
echo "  PID: ${PIDS[exp7_0.5B]}"

echo "[LAUNCH] Exp 7 — 1.5B on GPU 3"
$PYTHON "$SCRIPT_DIR/27_representation_injection.py" \
    --models 1.5B \
    --gpus 3 \
    > "$LOGDIR/exp7_1.5B.log" 2>&1 &
PIDS[exp7_1.5B]=$!
echo "  PID: ${PIDS[exp7_1.5B]}"

echo "[LAUNCH] Exp 7 — 3B on GPU 4"
$PYTHON "$SCRIPT_DIR/27_representation_injection.py" \
    --models 3B \
    --gpus 4 \
    > "$LOGDIR/exp7_3B.log" 2>&1 &
PIDS[exp7_3B]=$!
echo "  PID: ${PIDS[exp7_3B]}"

echo "[LAUNCH] Exp 7 — 7B on GPUs 5,6"
$PYTHON "$SCRIPT_DIR/27_representation_injection.py" \
    --models 7B \
    --gpus 5,6 \
    > "$LOGDIR/exp7_7B.log" 2>&1 &
PIDS[exp7_7B]=$!
echo "  PID: ${PIDS[exp7_7B]}"

echo ""
echo "All 5 tasks launched."
echo "PIDs:"
for name in "${!PIDS[@]}"; do
    echo "  $name: ${PIDS[$name]}"
done
echo ""

# ── Wait for Exp 7 smaller models first ──
echo "=== Waiting for Exp 7 (0.5B, 1.5B, 3B) ==="
FAILED=0
for name in exp7_0.5B exp7_1.5B exp7_3B; do
    pid=${PIDS[$name]}
    wait "$pid"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] $name (PID $pid) exited with code $rc at $(date)"
        FAILED=$((FAILED + 1))
    else
        echo "[OK]   $name completed at $(date)"
    fi
done

echo ""
echo "=== Waiting for Exp 7 — 7B ==="
wait ${PIDS[exp7_7B]}
rc=$?
if [ $rc -ne 0 ]; then
    echo "[FAIL] exp7_7B exited with code $rc at $(date)"
    FAILED=$((FAILED + 1))
else
    echo "[OK]   exp7_7B completed at $(date)"
fi

# ── Plot aggregation ──
echo ""
echo "[RUN] Exp 7 cross-scale plots..."
$PYTHON "$SCRIPT_DIR/27_representation_injection.py" \
    --only-plot \
    --models "0.5B,1.5B,3B,7B" \
    > "$LOGDIR/exp7_plot.log" 2>&1
rc=$?
[ $rc -eq 0 ] && echo "[OK]   Exp 7 plots generated"
[ $rc -ne 0 ] && echo "[FAIL] Exp 7 plots (code $rc)"

# ── Wait for 7B Exp 1-2 ──
echo ""
echo "=== Waiting for 7B Exp 1-2 ==="
wait ${PIDS[exp1_7B]}
rc=$?
if [ $rc -ne 0 ]; then
    echo "[FAIL] exp1_7B exited with code $rc at $(date)"
    FAILED=$((FAILED + 1))
else
    echo "[OK]   exp1_7B completed at $(date)"
fi

# ── Final Exp 1-2 aggregation ──
echo ""
echo "[RUN] Exp 1-2 final plot aggregation..."
$PYTHON "$SCRIPT_DIR/23_good_prefix_injection.py" \
    --only-plot \
    --models "0.5B,1.5B,3B,7B" \
    > "$LOGDIR/exp1_final_plot.log" 2>&1
rc=$?
[ $rc -eq 0 ] && echo "[OK]   Exp 1-2 plots generated"
[ $rc -ne 0 ] && echo "[FAIL] Exp 1-2 plots (code $rc)"

echo ""
echo "============================================================"
echo "  ALL DONE at $(date)"
echo "  Failed: $FAILED"
echo "============================================================"
echo "Results:"
echo "  Exp 7: $PROJECT_ROOT/runs/representation_injection/"
echo "  Exp 1-2: $PROJECT_ROOT/runs/good_prefix_exp/"
echo "  Logs: $LOGDIR/"
