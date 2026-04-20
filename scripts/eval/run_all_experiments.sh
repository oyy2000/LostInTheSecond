#!/bin/bash
# =============================================================================
# Full experiment pipeline v2 — prefix correction + corruption
#
# Run stages sequentially. Each stage can be re-run independently.
# Set environment variables before running:
#   export OPENAI_API_KEY=...
#   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#
# Usage:
#   bash scripts/eval/run_all_experiments.sh [stage]
#   stage: 1=sample, 2a=correct, 2b=corrupt, 3a=prefill_corr, 3b=prefill_corrupt,
#          4=prm, 5=hidden, 6=probes, 7a=did_corr, 7b=did_corrupt, 8=ig, 9=stats, 10=plots
#          all=run everything (default)
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."

RESULTS="results/gsm8k_7b_v2"
COT_FILE="${RESULTS}/raw_cot_n8.jsonl"
STAGE="${1:-all}"

GPUS="0,1,2,3,4,5,6,7"

echo "============================================"
echo "  Prefix Correction Experiment v2 Pipeline"
echo "  Model: Qwen2.5-7B-Instruct"
echo "  Results dir: ${RESULTS}"
echo "  GPUs: ${GPUS}"
echo "  Stage: ${STAGE}"
echo "============================================"

mkdir -p "${RESULTS}"

# ---------------------------------------------------------------------------
# Stage 1: Sample N=8 CoT trajectories
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "1" ]]; then
    echo ""
    echo ">>> Stage 1: Sampling CoT trajectories..."
    python scripts/data_prep/sample_multi_cot.py \
        --out-file "${COT_FILE}" \
        --n-samples 8 \
        --min-steps 4 \
        --gpus "${GPUS}"
fi

# ---------------------------------------------------------------------------
# Stage 2a: GPT incremental correction (wrong trajectories)
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "2a" ]]; then
    echo ""
    echo ">>> Stage 2a: GPT incremental correction..."
    python scripts/data_prep/construct_corrected_prefix_by_gpt.py \
        --in-file "${COT_FILE}" \
        --out-dir "${RESULTS}" \
        --k-values 1,2,3,4 \
        --workers 8 \
        --resume
fi

# ---------------------------------------------------------------------------
# Stage 2b: GPT incremental corruption (correct trajectories)
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "2b" ]]; then
    echo ""
    echo ">>> Stage 2b: GPT incremental corruption..."
    python scripts/data_prep/construct_corrupted_prefix_by_gpt.py \
        --in-file "${COT_FILE}" \
        --out-dir "${RESULTS}" \
        --k-values 1,2,3,4 \
        --workers 8 \
        --resume
fi

# ---------------------------------------------------------------------------
# Stage 3a: Prefill corrected prefix + generate tail
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "3a" ]]; then
    echo ""
    echo ">>> Stage 3a: Prefill corrected tails..."
    python scripts/data_prep/prefill_corrected_tail.py \
        --correction-dir "${RESULTS}" \
        --mode corrected \
        --k-values 1,2,3,4 \
        --gpus "${GPUS}"
fi

# ---------------------------------------------------------------------------
# Stage 3b: Prefill corrupted prefix + generate tail
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "3b" ]]; then
    echo ""
    echo ">>> Stage 3b: Prefill corrupted tails..."
    python scripts/data_prep/prefill_corrected_tail.py \
        --correction-dir "${RESULTS}" \
        --mode corrupted \
        --k-values 1,2,3,4 \
        --gpus "${GPUS}"
fi

# ---------------------------------------------------------------------------
# Stage 4: PRM scoring
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "4" ]]; then
    echo ""
    echo ">>> Stage 4: PRM scoring..."
    python scripts/eval/score_all_trajectories.py \
        --cot-file "${COT_FILE}" \
        --correction-dir "${RESULTS}" \
        --out-file "${RESULTS}/prm_scores_all.jsonl" \
        --gpus auto
fi

# ---------------------------------------------------------------------------
# Stage 5: Extract hidden states
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "5" ]]; then
    echo ""
    echo ">>> Stage 5: Extracting hidden states..."
    python scripts/eval/extract_step_hidden_states.py \
        --cot-file "${COT_FILE}" \
        --correction-dir "${RESULTS}" \
        --out-dir "${RESULTS}/hidden_states" \
        --gpus auto \
        --max-samples 500
fi

# ---------------------------------------------------------------------------
# Stage 6: Train probes
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "6" ]]; then
    echo ""
    echo ">>> Stage 6: Training probes..."
    python scripts/eval/train_step_probes.py \
        --hs-dir "${RESULTS}/hidden_states" \
        --out-dir "${RESULTS}/probes"
fi

# ---------------------------------------------------------------------------
# Stage 7a: DID analysis (corrected)
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "7a" ]]; then
    echo ""
    echo ">>> Stage 7a: DID analysis (corrected)..."
    for k in 1 2 3 4; do
        python scripts/eval/diff_in_diff_probe.py \
            --hs-dir "${RESULTS}/hidden_states" \
            --out-dir "${RESULTS}/diff_in_diff" \
            --modified-condition "corrected_k${k}"
    done
fi

# ---------------------------------------------------------------------------
# Stage 7b: DID analysis (corrupted)
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "7b" ]]; then
    echo ""
    echo ">>> Stage 7b: DID analysis (corrupted)..."
    for k in 1 2 3 4; do
        python scripts/eval/diff_in_diff_probe.py \
            --hs-dir "${RESULTS}/hidden_states" \
            --out-dir "${RESULTS}/diff_in_diff" \
            --modified-condition "corrupted_k${k}"
    done
fi

# ---------------------------------------------------------------------------
# Stage 8: Information Gain
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "8" ]]; then
    echo ""
    echo ">>> Stage 8: Computing information gain..."
    python scripts/eval/compute_step_ig.py \
        --cot-file "${COT_FILE}" \
        --correction-dir "${RESULTS}" \
        --out-dir "${RESULTS}/information_gain" \
        --gpus auto \
        --max-samples 300
fi

# ---------------------------------------------------------------------------
# Stage 9: Statistical tests
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "9" ]]; then
    echo ""
    echo ">>> Stage 9: Statistical tests..."
    python scripts/eval/statistical_tests.py \
        --cot-file "${COT_FILE}" \
        --correction-dir "${RESULTS}" \
        --out-dir "${RESULTS}/statistics"
fi

# ---------------------------------------------------------------------------
# Stage 10: Paper figures
# ---------------------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "10" ]]; then
    echo ""
    echo ">>> Stage 10: Generating paper figures..."
    python scripts/eval/plot_paper_figures.py \
        --results-root "${RESULTS}" \
        --cot-file "${COT_FILE}" \
        --out-dir figures/prefix_correction_v2
fi

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
