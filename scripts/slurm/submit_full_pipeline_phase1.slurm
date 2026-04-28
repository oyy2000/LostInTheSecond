#!/bin/bash
#SBATCH --job-name=lits-p1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/jet/home/swang47/yang/projects/LostInTheSecond/logs/phase1-%j.out
#SBATCH --error=/jet/home/swang47/yang/projects/LostInTheSecond/logs/phase1-%j.err

set -euo pipefail

module load cuda/12.6.1
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

export HF_TOKEN=$(cat /jet/home/swang47/.cache/huggingface/token)
export PYTHONUNBUFFERED=1

cd /jet/home/swang47/yang/projects/LostInTheSecond
ARTIFACTS=./artifacts_real/full

mkdir -p "$ARTIFACTS"

echo "================================================"
echo "  Phase 1: Sample ALL questions + LLaMA-3 generation"
echo "  Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Started: $(date)"
echo "================================================"

# ── Step 1: Sample ALL unique questions from LEMMA ───────────
QUESTIONS="$ARTIFACTS/lemma_all_questions.jsonl"
if [ ! -f "$QUESTIONS" ]; then
    echo ""
    echo "=== Step 1: Extract all unique questions from LEMMA ==="
    python3 scripts/data_prep/10_sample_lemma_questions.py \
        --n-samples -1 \
        --out-file "$QUESTIONS"
else
    echo "=== Step 1: Questions file exists, skipping ==="
    wc -l "$QUESTIONS"
fi

# ── Step 2: Generate with LLaMA-3-8B-Instruct (vLLM) ────────
GENERATIONS="$ARTIFACTS/lemma_llama3_generations.json"
if [ ! -f "$GENERATIONS" ]; then
    echo ""
    echo "=== Step 2: Generate solutions with LLaMA-3-8B-Instruct ==="
    python3 scripts/data_prep/11_generate_with_llama3.py \
        --in-file "$QUESTIONS" \
        --out-file "$GENERATIONS" \
        --limit 0
    echo "  -> $GENERATIONS"
else
    echo "=== Step 2: Generations file exists, skipping ==="
fi

echo ""
echo "================================================"
echo "  Phase 1 complete!"
echo "  Questions: $QUESTIONS"
echo "  Generations: $GENERATIONS"
echo "  Ended: $(date)"
echo "================================================"
