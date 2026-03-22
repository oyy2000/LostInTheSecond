#!/bin/bash
#SBATCH --job-name=lemma-pipe
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/jet/home/swang47/yang/projects/LostInTheSecond/logs/pipeline-%j.out
#SBATCH --error=/jet/home/swang47/yang/projects/LostInTheSecond/logs/pipeline-%j.err

set -euo pipefail

module load cuda/12.6.1
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

export HF_TOKEN=$(cat /jet/home/swang47/.cache/huggingface/token)
export PYTHONUNBUFFERED=1

cd /jet/home/swang47/yang/projects/LostInTheSecond
ARTIFACTS="./artifacts_real"

echo "================================================"
echo "  LEMMA Pipeline - Full (1000 samples)"
echo "  Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Started: $(date)"
echo "================================================"

nvidia-smi

# ── Step 1: Questions already sampled ────────────────────────
QUESTIONS="$ARTIFACTS/lemma_sampled_questions.jsonl"
if [ ! -f "$QUESTIONS" ]; then
    echo "=== Step 1: Sample questions ==="
    python3 scripts/data_prep/10_sample_lemma_questions.py \
        --n-samples 1000 \
        --out-file "$QUESTIONS"
fi
echo "Questions file: $QUESTIONS ($(wc -l < "$QUESTIONS") lines)"

# ── Step 2: Generate with LLaMA-3-8B-Instruct ───────────────
echo ""
echo "=== Step 2: Generate solutions with LLaMA-3-8B-Instruct (vLLM) ==="
GENERATIONS="$ARTIFACTS/lemma_llama3_generations.json"

python3 scripts/data_prep/11_generate_with_llama3.py \
    --in-file "$QUESTIONS" \
    --out-file "$GENERATIONS" \
    --use-vllm

echo "Generations: $GENERATIONS"

# ── Step 3: GPT Step 2 correction ───────────────────────────
echo ""
echo "=== Step 3: GPT Step 2 correction ==="
FIX_FILE="$ARTIFACTS/lemma_ds2_fix_step2_gpt.json"
WAIT_FILE="$ARTIFACTS/lemma_ds2_wait_recompute_gpt.json"

python3 scripts/data_prep/12_gpt_fix_step2_lemma.py \
    --in-file "$GENERATIONS" \
    --out-fix "$FIX_FILE" \
    --out-wait "$WAIT_FILE" \
    --audit-json "$ARTIFACTS/lemma_gpt_fix_step2_audit.json" \
    --judge-first \
    --model gpt-4o-mini

echo "Fix file:  $FIX_FILE"
echo "Wait file: $WAIT_FILE"

# ── Step 4: Prefill continuation ────────────────────────────
echo ""
echo "=== Step 4a: Prefill fix_step2 variant ==="
PREFILL_FIX="$ARTIFACTS/lemma_ds2_fix_step2_gpt_prefill.json"

python3 scripts/data_prep/13_prefill_llama3.py \
    --in-file "$FIX_FILE" \
    --out-file "$PREFILL_FIX" \
    --keep-steps 2 \
    --max-attempts 5

echo ""
echo "=== Step 4b: Prefill wait_recompute variant ==="
PREFILL_WAIT="$ARTIFACTS/lemma_ds2_wait_recompute_gpt_prefill.json"

python3 scripts/data_prep/13_prefill_llama3.py \
    --in-file "$WAIT_FILE" \
    --out-file "$PREFILL_WAIT" \
    --keep-steps 0 \
    --max-attempts 5

# ── Step 5: Convert to LEMMA format ─────────────────────────
echo ""
echo "=== Step 5: Convert to LEMMA training format ==="

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$PREFILL_FIX" \
    --out-file "$ARTIFACTS/lemma_sft_fix_step2.json"

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$PREFILL_WAIT" \
    --out-file "$ARTIFACTS/lemma_sft_wait_recompute.json"

# also create a version without correctness requirement
python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$PREFILL_WAIT" \
    --out-file "$ARTIFACTS/lemma_sft_wait_recompute_all.json" \
    --no-require-correct

echo ""
echo "================================================"
echo "  Pipeline complete!"
echo "  Ended: $(date)"
echo "================================================"
echo ""
echo "Output files:"
ls -lh "$ARTIFACTS"/lemma_sft_*.json
