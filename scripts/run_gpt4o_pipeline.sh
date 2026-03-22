#!/bin/bash
set -e

source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

cd /jet/home/swang47/yang/projects/LostInTheSecond

export OPENAI_API_KEY="sk-sNqUCd2XmGHrzXoZDCTmLa7iHdi9L5paDJhTpyNfwkHW8Hc8"
export OPENAI_BASE_URL="https://yinli.one/v1"

OUT_DIR="artifacts_real/full_gpt4o"
mkdir -p "$OUT_DIR"

echo "============================================="
echo "  Phase 1: GPT-4o Step 2 Fix"
echo "  Started: $(date)"
echo "============================================="

python scripts/data_prep/12_gpt_fix_step2_lemma.py \
    --in-file artifacts_real/full/lemma_llama3_generations.json \
    --out-fix "$OUT_DIR/lemma_ds2_fix_step2_gpt4o.json" \
    --out-wait "$OUT_DIR/lemma_ds2_wait_recompute_gpt4o.json" \
    --audit-json "$OUT_DIR/lemma_gpt4o_fix_step2_audit.json" \
    --model gpt-4o \
    --temperature 0.0 \
    --only-incorrect

echo ""
echo "============================================="
echo "  Phase 2: GPT-4o Step 3 Fix"
echo "  Started: $(date)"
echo "============================================="

python scripts/data_prep/20_gpt_fix_step3_only.py \
    --in-file artifacts_real/full/lemma_llama3_generations.json \
    --step2-fix-file "$OUT_DIR/lemma_ds2_fix_step2_gpt4o.json" \
    --out-fix "$OUT_DIR/lemma_ds2_fix_step3_gpt4o.json" \
    --out-wait "$OUT_DIR/lemma_ds2_wait_step3_gpt4o.json" \
    --model gpt-4o \
    --temperature 0.0

echo ""
echo "============================================="
echo "  Phase 1+2 Completed: $(date)"
echo "============================================="
echo ""
echo "Step 2 fix: $(python -c "import json; d=json.load(open('$OUT_DIR/lemma_ds2_fix_step2_gpt4o.json')); print(len(d.get('samples',d)))")"
echo "Step 2 wait: $(python -c "import json; d=json.load(open('$OUT_DIR/lemma_ds2_wait_recompute_gpt4o.json')); print(len(d.get('samples',d)))")"
echo "Step 3 fix: $(python -c "import json; d=json.load(open('$OUT_DIR/lemma_ds2_fix_step3_gpt4o.json')); print(len(d.get('samples',d)))")"
echo "Step 3 wait: $(python -c "import json; d=json.load(open('$OUT_DIR/lemma_ds2_wait_step3_gpt4o.json')); print(len(d.get('samples',d)))")"
echo ""
echo "GPT-4o correction phases complete. Next: prefill + SFT build + training."
