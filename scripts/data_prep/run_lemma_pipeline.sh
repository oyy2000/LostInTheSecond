#!/bin/bash
set -euo pipefail

# ============================================================
# LostInTheSecond → LEMMA pipeline
#
# Usage:
#   bash scripts/data_prep/run_lemma_pipeline.sh [small|full]
#
# "small" (default): 10 samples for validation
# "full":            1000 samples
# ============================================================

MODE="${1:-small}"
cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

echo "================================================"
echo "  LostInTheSecond LEMMA Pipeline  (mode=$MODE)"
echo "  Project: $PROJECT_ROOT"
echo "================================================"

if [ "$MODE" = "small" ]; then
    N_SAMPLE=10
    GEN_LIMIT=10
    GPT_LIMIT=10
    PREFILL_LIMIT=10
    SUFFIX="_small"
elif [ "$MODE" = "full" ]; then
    N_SAMPLE=1000
    GEN_LIMIT=0
    GPT_LIMIT=0
    PREFILL_LIMIT=0
    SUFFIX=""
else
    echo "Usage: $0 [small|full]"
    exit 1
fi

ARTIFACTS="$PROJECT_ROOT/artifacts_real"
mkdir -p "$ARTIFACTS"

# ── Step 1: Sample questions from LEMMA ──────────────────────
echo ""
echo "=== Step 1: Sample $N_SAMPLE questions from LEMMA ==="
QUESTIONS="$ARTIFACTS/lemma_sampled_questions${SUFFIX}.jsonl"

python3 scripts/data_prep/10_sample_lemma_questions.py \
    --n-samples "$N_SAMPLE" \
    --out-file "$QUESTIONS"

echo "  -> $QUESTIONS"

# ── Step 2: Generate with LLaMA-3-8B-Instruct ───────────────
echo ""
echo "=== Step 2: Generate solutions with LLaMA-3-8B-Instruct ==="
GENERATIONS="$ARTIFACTS/lemma_llama3_generations${SUFFIX}.json"

python3 scripts/data_prep/11_generate_with_llama3.py \
    --in-file "$QUESTIONS" \
    --out-file "$GENERATIONS" \
    --limit "$GEN_LIMIT"

echo "  -> $GENERATIONS"

# ── Step 3: GPT Fix Step 2 ──────────────────────────────────
echo ""
echo "=== Step 3: GPT Step 2 correction ==="
FIX_FILE="$ARTIFACTS/lemma_ds2_fix_step2_gpt${SUFFIX}.json"
WAIT_FILE="$ARTIFACTS/lemma_ds2_wait_recompute_gpt${SUFFIX}.json"
AUDIT_FILE="$ARTIFACTS/lemma_gpt_fix_step2_audit${SUFFIX}.json"

python3 scripts/data_prep/12_gpt_fix_step2_lemma.py \
    --in-file "$GENERATIONS" \
    --out-fix "$FIX_FILE" \
    --out-wait "$WAIT_FILE" \
    --audit-json "$AUDIT_FILE" \
    --limit "$GPT_LIMIT" \
    --judge-first

echo "  -> $FIX_FILE"
echo "  -> $WAIT_FILE"

# ── Step 4: Prefill continuation (wait+recompute variant) ───
echo ""
echo "=== Step 4: Prefill continuation ==="
PREFILL_FIX="$ARTIFACTS/lemma_ds2_fix_step2_gpt_prefill${SUFFIX}.json"
PREFILL_WAIT="$ARTIFACTS/lemma_ds2_wait_recompute_gpt_prefill${SUFFIX}.json"

echo "  4a: Prefill fix_step2 variant..."
python3 scripts/data_prep/13_prefill_llama3.py \
    --in-file "$FIX_FILE" \
    --out-file "$PREFILL_FIX" \
    --keep-steps 2 \
    --limit "$PREFILL_LIMIT"

echo "  4b: Prefill wait_recompute variant..."
python3 scripts/data_prep/13_prefill_llama3.py \
    --in-file "$WAIT_FILE" \
    --out-file "$PREFILL_WAIT" \
    --keep-steps 0 \
    --limit "$PREFILL_LIMIT"

echo "  -> $PREFILL_FIX"
echo "  -> $PREFILL_WAIT"

# ── Step 5: Convert to LEMMA alpaca format ──────────────────
echo ""
echo "=== Step 5: Convert to LEMMA training format ==="
SFT_FIX="$ARTIFACTS/lemma_sft_fix_step2${SUFFIX}.json"
SFT_WAIT="$ARTIFACTS/lemma_sft_wait_recompute${SUFFIX}.json"

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$PREFILL_FIX" \
    --out-file "$SFT_FIX"

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$PREFILL_WAIT" \
    --out-file "$SFT_WAIT"

echo "  -> $SFT_FIX"
echo "  -> $SFT_WAIT"

echo ""
echo "================================================"
echo "  Pipeline complete!"
echo "  SFT data (fix_step2):       $SFT_FIX"
echo "  SFT data (wait_recompute):  $SFT_WAIT"
echo ""
echo "  Next: run LoRA or Full FT with these datasets."
echo "================================================"
