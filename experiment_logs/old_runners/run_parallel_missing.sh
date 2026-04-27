#!/bin/bash
set -uo pipefail

# ============================================================
#  Parallel launcher for all MISSING experiments
#  GPU 7: already running 3B Exp 1-2 (untouched)
#  GPU 0,1: 7B Exp 1-2 (2-GPU split)
#  GPU 2:   1.5B Exp 3 re-run (max_seq_len=512)
#  GPU 3:   3B Exp 3 re-run (max_seq_len=512)
#  GPU 4,5: 7B Exp 3 re-run (2-GPU split, max_seq_len=512)
#  GPU 6:   Exp 6 base vs instruct (CPU analysis, uses GPU briefly)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/common/home/sl2148/anaconda3/envs/fact_yang/bin/python"
LOGDIR="$PROJECT_ROOT/runs/parallel_logs"
mkdir -p "$LOGDIR"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES

echo "============================================================"
echo "  Parallel Missing Experiments (8 GPUs)"
echo "  Started: $(date)"
echo "============================================================"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
echo ""

declare -A PIDS

# ── Task 1: Exp 1-2 — 7B on GPUs 0,1 ──
echo "[LAUNCH] Exp 1-2 — 7B on GPUs 0,1"
$PYTHON "$SCRIPT_DIR/23_good_prefix_injection.py" \
    --models 7B \
    --gpus 0,1 \
    > "$LOGDIR/exp1_7B_2gpu.log" 2>&1 &
PIDS[exp1_7B]=$!
echo "  PID: ${PIDS[exp1_7B]}"

# ── Task 2: Exp 3 — 1.5B re-run on GPU 2 ──
rm -f "$PROJECT_ROOT/runs/hidden_state_mode/hidden_state_1.5B.json"
echo "[LAUNCH] Exp 3 — 1.5B re-run on GPU 2 (max_seq_len=512)"
$PYTHON "$SCRIPT_DIR/24_hidden_state_mode_analysis.py" \
    --models 1.5B \
    --gpus 2 \
    --max-seq-len 512 \
    > "$LOGDIR/exp3_1.5B_rerun.log" 2>&1 &
PIDS[exp3_1.5B]=$!
echo "  PID: ${PIDS[exp3_1.5B]}"

# ── Task 3: Exp 3 — 3B re-run on GPU 3 ──
rm -f "$PROJECT_ROOT/runs/hidden_state_mode/hidden_state_3B.json"
echo "[LAUNCH] Exp 3 — 3B re-run on GPU 3 (max_seq_len=512)"
$PYTHON "$SCRIPT_DIR/24_hidden_state_mode_analysis.py" \
    --models 3B \
    --gpus 3 \
    --max-seq-len 512 \
    > "$LOGDIR/exp3_3B_rerun.log" 2>&1 &
PIDS[exp3_3B]=$!
echo "  PID: ${PIDS[exp3_3B]}"

# ── Task 4: Exp 3 — 7B re-run on GPUs 4,5 ──
rm -f "$PROJECT_ROOT/runs/hidden_state_mode/hidden_state_7B.json"
echo "[LAUNCH] Exp 3 — 7B re-run on GPUs 4,5 (max_seq_len=512)"
$PYTHON "$SCRIPT_DIR/24_hidden_state_mode_analysis.py" \
    --models 7B \
    --gpus 4,5 \
    --max-seq-len 512 \
    > "$LOGDIR/exp3_7B_2gpu.log" 2>&1 &
PIDS[exp3_7B]=$!
echo "  PID: ${PIDS[exp3_7B]}"

echo ""
echo "All 4 GPU tasks launched. GPU 7 continues 3B Exp 1-2."
echo ""
echo "PIDs:"
for name in "${!PIDS[@]}"; do
    echo "  $name: ${PIDS[$name]}"
done
echo ""

# ── Wait for Exp 3 tasks (shorter) to finish first ──
echo "=== Waiting for Exp 3 re-runs to finish ==="
FAILED=0

for name in exp3_1.5B exp3_3B exp3_7B; do
    pid=${PIDS[$name]}
    wait "$pid"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[FAIL] $name (PID $pid) exited with code $rc at $(date)"
        FAILED=$((FAILED + 1))
    else
        echo "[OK]   $name (PID $pid) completed at $(date)"
    fi
done

echo ""
echo "Exp 3 re-runs done ($((3 - FAILED))/3 succeeded). $(date)"
echo ""

# ── Run Exp 3 cross-scale aggregation ──
echo "[RUN] Exp 3 plot aggregation..."
$PYTHON "$SCRIPT_DIR/24_hidden_state_mode_analysis.py" \
    --only-plot \
    --models "0.5B,1.5B,3B,7B" \
    > "$LOGDIR/exp3_plot_rerun.log" 2>&1
rc=$?
[ $rc -eq 0 ] && echo "[OK]   Exp 3 plots regenerated"
[ $rc -ne 0 ] && echo "[FAIL] Exp 3 plot aggregation (code $rc)"

# ── Run Exp 6: Base vs Instruct comparison (CPU-only) ──
echo ""
echo "=== Exp 6: Base vs Instruct comparison ==="

echo "[RUN] Base model summary..."
$PYTHON "$SCRIPT_DIR/22_multi_scale_baseline.py" \
    --only-plot \
    --models "0.5B-base,1.5B-base,3B-base,7B-base" \
    > "$LOGDIR/exp6_base_summary.log" 2>&1
rc=$?
[ $rc -eq 0 ] && echo "[OK]   Base model summary generated"
[ $rc -ne 0 ] && echo "[FAIL] Base model summary (code $rc)"

echo "[RUN] Combined instruct+base summary..."
$PYTHON "$SCRIPT_DIR/22_multi_scale_baseline.py" \
    --only-plot \
    --models "0.5B,1.5B,3B,7B,0.5B-base,1.5B-base,3B-base,7B-base" \
    > "$LOGDIR/exp6_combined_summary.log" 2>&1
rc=$?
[ $rc -eq 0 ] && echo "[OK]   Combined instruct+base summary generated"
[ $rc -ne 0 ] && echo "[FAIL] Combined summary (code $rc)"

# ── Re-run Exp 5 scaling law with updated data ──
echo ""
echo "[RUN] Exp 5 scaling law (re-fit with fresh Exp 3 data)..."
$PYTHON "$SCRIPT_DIR/26_scaling_law_mode_stability.py" \
    > "$LOGDIR/exp5_refit.log" 2>&1
rc=$?
[ $rc -eq 0 ] && echo "[OK]   Exp 5 scaling law updated"
[ $rc -ne 0 ] && echo "[FAIL] Exp 5 scaling law (code $rc)"

# ── Wait for Exp 1-2 7B (the longest task) ──
echo ""
echo "=== Waiting for Exp 1-2 — 7B to finish ==="
wait ${PIDS[exp1_7B]}
rc=$?
if [ $rc -ne 0 ]; then
    echo "[FAIL] exp1_7B (PID ${PIDS[exp1_7B]}) exited with code $rc at $(date)"
    FAILED=$((FAILED + 1))
else
    echo "[OK]   exp1_7B completed at $(date)"
fi

# ── Wait for 3B Exp 1-2 (on GPU 7) if still running ──
echo ""
echo "Checking if 3B Exp 1-2 (GPU 7) is still running..."
if pgrep -f "23_good_prefix_injection.py.*3B.*--gpus 7" > /dev/null 2>&1; then
    echo "[WAIT] 3B Exp 1-2 still running. Waiting..."
    while pgrep -f "23_good_prefix_injection.py.*3B" > /dev/null 2>&1; do
        sleep 60
        echo "  Still waiting... $(date)"
    done
    echo "[DONE] 3B Exp 1-2 finished at $(date)"
else
    echo "[DONE] 3B Exp 1-2 already finished"
fi

# ── Final aggregation for Exp 1-2 ──
echo ""
echo "=== Final Aggregation ==="

echo "[RUN] Exp 1-2 plot aggregation..."
$PYTHON "$SCRIPT_DIR/23_good_prefix_injection.py" \
    --only-plot \
    --models "0.5B,1.5B,3B,7B" \
    > "$LOGDIR/exp1_final_plot.log" 2>&1
rc=$?
[ $rc -eq 0 ] && echo "[OK]   Exp 1-2 plots generated"
[ $rc -ne 0 ] && echo "[FAIL] Exp 1-2 plot aggregation (code $rc)"

echo "[RUN] Exp 4 conditioned PRM (re-run)..."
$PYTHON "$SCRIPT_DIR/25_conditioned_prm_analysis.py" \
    --models "0.5B,1.5B,3B,7B" \
    > "$LOGDIR/exp4_final.log" 2>&1
rc=$?
[ $rc -eq 0 ] && echo "[OK]   Exp 4 conditioned PRM done"
[ $rc -ne 0 ] && echo "[FAIL] Exp 4 (code $rc)"

echo ""
echo "============================================================"
echo "  ALL DONE at $(date)"
echo "  Total failures: $FAILED"
echo "============================================================"
echo ""
echo "Results:"
echo "  Exp 1-2: $PROJECT_ROOT/runs/good_prefix_exp/"
echo "  Exp 3:   $PROJECT_ROOT/runs/hidden_state_mode/"
echo "  Exp 6:   $PROJECT_ROOT/runs/multi_scale_prm/"
echo "  Logs:    $LOGDIR/"
