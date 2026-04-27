#!/bin/bash
set -uo pipefail

# ============================================================
#  Autonomous Overnight Experiment Runner
#  Handles everything that remains:
#    1. Wait for current Exp 1-2 launcher to finish (0.5B, 1.5B, 3B)
#    2. Run 7B Exp 1-2 with 2-GPU split (GPUs 0,7)
#    3. Re-run Exp 3 7B with 2-GPU split + reduced max_seq_len
#    4. Generate Exp 6 base-vs-instruct summary & plots
#    5. Final aggregation & plotting for all experiments
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="/common/home/sl2148/anaconda3/envs/fact_yang/bin/python"
LOGDIR="$PROJECT_ROOT/runs/parallel_logs"
mkdir -p "$LOGDIR"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES

TOTAL_FAILED=0

echo "============================================================"
echo "  Autonomous Overnight Experiment Runner"
echo "  Started: $(date)"
echo "============================================================"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
echo ""

# ── Phase 1: Wait for current launcher to finish ──────────────
echo "=== Phase 1: Wait for current Exp 1-2 processes ==="

wait_for_process() {
    local pattern="$1"
    local desc="$2"
    while pgrep -f "$pattern" > /dev/null 2>&1; do
        local count
        count=$(pgrep -f "$pattern" 2>/dev/null | wc -l)
        echo "  [WAIT] $desc: $count process(es) still running... $(date)"
        sleep 120
    done
    echo "  [DONE] $desc finished at $(date)"
}

wait_for_process "23_good_prefix_injection.py" "Exp 1-2 (good-prefix injection)"
wait_for_process "run_parallel_exp1_3.sh" "Parallel launcher"

echo ""
echo "All prior experiment processes finished at $(date)"
echo ""
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
echo ""

# Brief pause for GPU memory release
sleep 30

# ── Phase 2: 7B Exp 1-2 with 2-GPU split ─────────────────────
echo "=== Phase 2: 7B Exp 1-2 with 2-GPU split (GPUs 0,7) ==="
echo "[RUN] $(date)"
$PYTHON "$SCRIPT_DIR/23_good_prefix_injection.py" \
    --models 7B \
    --gpus 0,7 \
    > "$LOGDIR/exp1_7B_2gpu.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    echo "[FAIL] 7B Exp 1-2 exited with code $rc"
    TOTAL_FAILED=$((TOTAL_FAILED + 1))
else
    echo "[OK]   7B Exp 1-2 completed at $(date)"
fi
echo ""

# ── Phase 3: Re-run Exp 3 for 7B (2 GPUs, smaller seq len) ───
echo "=== Phase 3: Re-run Exp 3 7B (2 GPUs, max_seq_len=512) ==="
rm -f "$PROJECT_ROOT/runs/hidden_state_mode/hidden_state_7B.json"
echo "[RUN] $(date)"
$PYTHON "$SCRIPT_DIR/24_hidden_state_mode_analysis.py" \
    --models 7B \
    --gpus 0,7 \
    --max-seq-len 512 \
    > "$LOGDIR/exp3_7B_2gpu.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    echo "[FAIL] Exp 3 7B re-run exited with code $rc"
    TOTAL_FAILED=$((TOTAL_FAILED + 1))
else
    echo "[OK]   Exp 3 7B re-run completed at $(date)"
fi
echo ""

# ── Phase 4: Exp 6 — Base vs Instruct comparison ─────────────
echo "=== Phase 4: Exp 6 — Base vs Instruct summary & plots ==="
echo "[RUN] Generating base model summary... $(date)"
$PYTHON "$SCRIPT_DIR/22_multi_scale_baseline.py" \
    --only-plot \
    --models "0.5B-base,1.5B-base,3B-base,7B-base" \
    > "$LOGDIR/exp6_base_summary.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    echo "[FAIL] Base model summary exited with code $rc"
    TOTAL_FAILED=$((TOTAL_FAILED + 1))
else
    echo "[OK]   Base model summary generated"
fi

echo "[RUN] Generating combined instruct+base summary... $(date)"
$PYTHON "$SCRIPT_DIR/22_multi_scale_baseline.py" \
    --only-plot \
    --models "0.5B,1.5B,3B,7B,0.5B-base,1.5B-base,3B-base,7B-base" \
    > "$LOGDIR/exp6_combined_summary.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    echo "[FAIL] Combined summary exited with code $rc"
    TOTAL_FAILED=$((TOTAL_FAILED + 1))
else
    echo "[OK]   Combined instruct+base summary generated"
fi
echo ""

# ── Phase 5: Final aggregation & plotting ─────────────────────
echo "=== Phase 5: Final aggregation ==="

echo "[RUN] Exp 1-2 plot aggregation..."
$PYTHON "$SCRIPT_DIR/23_good_prefix_injection.py" \
    --only-plot \
    --models "0.5B,1.5B,3B,7B" \
    > "$LOGDIR/final_exp1_plot.log" 2>&1
rc=$?
[ $rc -ne 0 ] && echo "[FAIL] Exp 1-2 plot (code $rc)" && TOTAL_FAILED=$((TOTAL_FAILED + 1))
[ $rc -eq 0 ] && echo "[OK]   Exp 1-2 plots generated"

echo "[RUN] Exp 3 plot aggregation..."
$PYTHON "$SCRIPT_DIR/24_hidden_state_mode_analysis.py" \
    --only-plot \
    --models "0.5B,1.5B,3B,7B" \
    > "$LOGDIR/final_exp3_plot.log" 2>&1
rc=$?
[ $rc -ne 0 ] && echo "[FAIL] Exp 3 plot (code $rc)" && TOTAL_FAILED=$((TOTAL_FAILED + 1))
[ $rc -eq 0 ] && echo "[OK]   Exp 3 plots generated"

echo "[RUN] Exp 4 conditioned PRM analysis..."
$PYTHON "$SCRIPT_DIR/25_conditioned_prm_analysis.py" \
    --models "0.5B,1.5B,3B,7B" \
    > "$LOGDIR/final_exp4.log" 2>&1
rc=$?
[ $rc -ne 0 ] && echo "[FAIL] Exp 4 (code $rc)" && TOTAL_FAILED=$((TOTAL_FAILED + 1))
[ $rc -eq 0 ] && echo "[OK]   Exp 4 conditioned analysis done"

echo "[RUN] Exp 5 scaling law analysis..."
$PYTHON "$SCRIPT_DIR/26_scaling_law_mode_stability.py" \
    > "$LOGDIR/final_exp5.log" 2>&1
rc=$?
[ $rc -ne 0 ] && echo "[FAIL] Exp 5 (code $rc)" && TOTAL_FAILED=$((TOTAL_FAILED + 1))
[ $rc -eq 0 ] && echo "[OK]   Exp 5 scaling law analysis done"

echo ""
echo "============================================================"
echo "  ALL DONE at $(date)"
echo "  Failed: $TOTAL_FAILED"
echo "============================================================"
echo ""
echo "Results:"
echo "  Exp 0 (baselines):  $PROJECT_ROOT/runs/multi_scale_prm/"
echo "  Exp 1-2 (prefix):   $PROJECT_ROOT/runs/good_prefix_exp/"
echo "  Exp 3 (hidden):     $PROJECT_ROOT/runs/hidden_state_mode/"
echo "  Exp 4 (conditioned):$PROJECT_ROOT/runs/conditioned_prm/"
echo "  Exp 5 (scaling):    $PROJECT_ROOT/runs/scaling_law_analysis/"
echo "  Logs:               $LOGDIR/"
echo ""
echo "Check logs for details:"
echo "  tail $LOGDIR/exp1_7B_2gpu.log"
echo "  tail $LOGDIR/exp3_7B_2gpu.log"
echo "  tail $LOGDIR/exp6_base_summary.log"
