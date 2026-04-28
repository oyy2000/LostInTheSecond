#!/bin/bash
#SBATCH --job-name=prefill-lm
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/jet/home/swang47/yang/projects/LostInTheSecond/logs/prefill-%j.out
#SBATCH --error=/jet/home/swang47/yang/projects/LostInTheSecond/logs/prefill-%j.err

set -euo pipefail

module load cuda/12.6.1
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

export HF_TOKEN=$(cat /jet/home/swang47/.cache/huggingface/token)
export PYTHONUNBUFFERED=1

cd /jet/home/swang47/yang/projects/LostInTheSecond
ARTIFACTS="./artifacts_real"

echo "================================================"
echo "  Prefill + Convert Steps (restarted)"
echo "  Node: $(hostname)"
echo "  Started: $(date)"
echo "================================================"
nvidia-smi

# ── Step 4a: Prefill fix_step2 ───────────────────────────────
PREFILL_FIX="$ARTIFACTS/lemma_ds2_fix_step2_gpt_prefill.json"
if [ ! -f "$PREFILL_FIX" ]; then
    echo "=== Step 4a: Prefill fix_step2 variant ==="
    python3 scripts/data_prep/13_prefill_llama3.py \
        --in-file "$ARTIFACTS/lemma_ds2_fix_step2_gpt.json" \
        --out-file "$PREFILL_FIX" \
        --keep-steps 2 \
        --max-attempts 5
    echo "  -> $PREFILL_FIX"
else
    echo "=== Step 4a: Already exists, skipping ==="
fi

# ── Step 4b: Prefill wait_recompute ──────────────────────────
PREFILL_WAIT="$ARTIFACTS/lemma_ds2_wait_recompute_gpt_prefill.json"
if [ ! -f "$PREFILL_WAIT" ]; then
    echo "=== Step 4b: Prefill wait_recompute variant ==="
    python3 scripts/data_prep/13_prefill_llama3.py \
        --in-file "$ARTIFACTS/lemma_ds2_wait_recompute_gpt.json" \
        --out-file "$PREFILL_WAIT" \
        --keep-steps 0 \
        --max-attempts 5
    echo "  -> $PREFILL_WAIT"
else
    echo "=== Step 4b: Already exists, skipping ==="
fi

# ── Step 5: Convert to LEMMA format ─────────────────────────
echo "=== Step 5: Convert to LEMMA training format ==="

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$PREFILL_FIX" \
    --out-file "$ARTIFACTS/lemma_sft_fix_step2.json"

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$PREFILL_WAIT" \
    --out-file "$ARTIFACTS/lemma_sft_wait_recompute.json"

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$PREFILL_WAIT" \
    --out-file "$ARTIFACTS/lemma_sft_wait_recompute_all.json" \
    --no-require-correct

echo ""
echo "================================================"
echo "  Prefill + Convert complete!"
echo "  Ended: $(date)"
echo "================================================"
ls -lh "$ARTIFACTS"/lemma_sft_*.json
