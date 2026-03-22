#!/bin/bash
#SBATCH --job-name=lits-p3
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/jet/home/swang47/yang/projects/LostInTheSecond/logs/phase3-%j.out
#SBATCH --error=/jet/home/swang47/yang/projects/LostInTheSecond/logs/phase3-%j.err

set -euo pipefail

module load cuda/12.6.1
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

export HF_TOKEN=$(cat /jet/home/swang47/.cache/huggingface/token)
export PYTHONUNBUFFERED=1

cd /jet/home/swang47/yang/projects/LostInTheSecond
ARTIFACTS=./artifacts_real/full
MODEL_DIR=/ocean/projects/cis250050p/swang47/yang/lostinsecond-models/full_run

mkdir -p "$ARTIFACTS" "$MODEL_DIR"

echo "================================================"
echo "  Phase 3: Prefill + Convert + FT"
echo "  Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Started: $(date)"
echo "================================================"

FIX_FILE="$ARTIFACTS/lemma_ds2_fix_step2_gpt.json"
WAIT_FILE="$ARTIFACTS/lemma_ds2_wait_recompute_gpt.json"

if [ ! -f "$FIX_FILE" ] || [ ! -f "$WAIT_FILE" ]; then
    echo "ERROR: Phase 2 outputs not found"
    exit 1
fi

# ── Step 4a: Prefill fix_step2 ───────────────────────────────
PREFILL_FIX="$ARTIFACTS/lemma_ds2_fix_step2_gpt_prefill.json"
if [ ! -f "$PREFILL_FIX" ]; then
    echo ""
    echo "=== Step 4a: Prefill fix_step2 ==="
    python3 scripts/data_prep/13_prefill_llama3.py \
        --in-file "$FIX_FILE" \
        --out-file "$PREFILL_FIX" \
        --keep-steps 2 \
        --limit 0
else
    echo "=== Step 4a: Prefill fix exists, skipping ==="
fi

# ── Step 4b: Prefill wait_recompute ──────────────────────────
PREFILL_WAIT="$ARTIFACTS/lemma_ds2_wait_recompute_gpt_prefill.json"
if [ ! -f "$PREFILL_WAIT" ]; then
    echo ""
    echo "=== Step 4b: Prefill wait_recompute ==="
    python3 scripts/data_prep/13_prefill_llama3.py \
        --in-file "$WAIT_FILE" \
        --out-file "$PREFILL_WAIT" \
        --keep-steps 0 \
        --limit 0
else
    echo "=== Step 4b: Prefill wait exists, skipping ==="
fi

# ── Step 5: Convert to LEMMA format ─────────────────────────
SFT_FIX="$ARTIFACTS/lemma_sft_fix_step2.json"
SFT_WAIT="$ARTIFACTS/lemma_sft_wait_recompute.json"
SFT_WAIT_ALL="$ARTIFACTS/lemma_sft_wait_recompute_all.json"

echo ""
echo "=== Step 5: Convert to LEMMA training format ==="

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$PREFILL_FIX" \
    --out-file "$SFT_FIX"

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$PREFILL_WAIT" \
    --out-file "$SFT_WAIT"

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$PREFILL_WAIT" \
    --out-file "$SFT_WAIT_ALL" \
    --no-require-correct

echo ""
echo "SFT datasets created:"
ls -lh "$ARTIFACTS"/lemma_sft_*.json

# ── Step 6: LoRA fine-tune ───────────────────────────────────
echo ""
echo "=== Step 6: LoRA fine-tune ==="
LORA_OUT="$MODEL_DIR/lora_llama3_8b"

python3 scripts/data_prep/15_finetune_lemma.py \
    --method lora \
    --dataset-path "$SFT_WAIT_ALL" \
    --output-dir "$LORA_OUT" \
    --lr 1e-4 \
    --warmup-ratio 0.03 \
    --epochs 3 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 8

echo ""
echo "================================================"
echo "  Phase 3 complete!"
echo "  LoRA model: $LORA_OUT"
echo "  Ended: $(date)"
echo "================================================"
