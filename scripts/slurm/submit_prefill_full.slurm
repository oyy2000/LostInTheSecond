#!/bin/bash
#SBATCH --job-name=prefill-full
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --account=cis250050p
#SBATCH --output=/jet/home/swang47/yang/projects/LostInTheSecond/logs/prefill-full-%j.out
#SBATCH --error=/jet/home/swang47/yang/projects/LostInTheSecond/logs/prefill-full-%j.err

set -euo pipefail

module load cuda/12.6.1
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

export HF_TOKEN=$(cat /jet/home/swang47/.cache/huggingface/token)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd /jet/home/swang47/yang/projects/LostInTheSecond
ARTIFACTS="./artifacts_real/full"
TP=2

echo "================================================"
echo "  Prefill + Convert (FULL dataset, vLLM, tp=$TP)"
echo "  Node: $(hostname)"
echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "  Started: $(date)"
echo "================================================"
nvidia-smi

# ── Step 1: Prefill fix_step2 variant ────────────────────────
PREFILL_FIX="$ARTIFACTS/lemma_ds2_fix_step2_gpt_prefill.json"
echo ""
echo "=== Prefill fix_step2 variant (vLLM, tp=$TP) ==="
python3 -u scripts/data_prep/13_prefill_llama3.py \
    --in-file "$ARTIFACTS/lemma_ds2_fix_step2_gpt.json" \
    --out-file "$PREFILL_FIX" \
    --keep-steps 2 \
    --max-attempts 5 \
    --use-vllm \
    --tensor-parallel $TP \
    --dtype float16
echo "  -> $PREFILL_FIX"

# ── Step 2: Prefill wait_recompute variant ───────────────────
PREFILL_WAIT="$ARTIFACTS/lemma_ds2_wait_recompute_gpt_prefill.json"
echo ""
echo "=== Prefill wait_recompute variant (vLLM, tp=$TP) ==="
python3 -u scripts/data_prep/13_prefill_llama3.py \
    --in-file "$ARTIFACTS/lemma_ds2_wait_recompute_gpt.json" \
    --out-file "$PREFILL_WAIT" \
    --keep-steps 0 \
    --max-attempts 5 \
    --use-vllm \
    --tensor-parallel $TP \
    --dtype float16
echo "  -> $PREFILL_WAIT"

# ── Step 3: Convert to LEMMA training format ─────────────────
echo ""
echo "=== Convert to LEMMA training format ==="

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
