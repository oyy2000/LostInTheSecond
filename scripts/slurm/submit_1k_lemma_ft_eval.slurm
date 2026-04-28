#!/bin/bash
#SBATCH --job-name=1k-ft-ev
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/jet/home/swang47/yang/projects/LostInTheSecond/logs/1k-ft-eval-%j.out
#SBATCH --error=/jet/home/swang47/yang/projects/LostInTheSecond/logs/1k-ft-eval-%j.err

set -euo pipefail

module load cuda/12.6.1
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

export HF_TOKEN=$(cat /jet/home/swang47/.cache/huggingface/token)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd /jet/home/swang47/yang/projects/LostInTheSecond

DATASET=./artifacts_real/lemma_1k_original_sft.json
MODEL_OUT=/ocean/projects/cis250050p/swang47/yang/lostinsecond-models/full_ft_1k_lemma_original
EVAL_DIR=/jet/home/swang47/yang/projects/LEMMA/evaluation

echo "================================================"
echo "  1K LEMMA Original SFT -> Full FT -> Eval"
echo "  Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Started: $(date)"
echo "================================================"
nvidia-smi

echo ""
echo "Dataset: $DATASET"
python3 -c "import json; d=json.load(open('$DATASET')); print(f'  Entries: {len(d)}')"

# ── Step 1: Full Fine-tune ───────────────────────────────────
echo ""
echo "=== Step 1: Full Fine-tune (LLaMA-3-8B, 1xH100) ==="

python3 scripts/data_prep/15_finetune_lemma.py \
    --method full \
    --dataset-path "$DATASET" \
    --output-dir "$MODEL_OUT" \
    --learning-rate 1e-5 \
    --warmup-ratio 0.03 \
    --num-train-epochs 3 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 8 \
    --max-length 2048

echo ""
echo "=== Step 1 complete. Model saved to: $MODEL_OUT ==="

# ── Step 2: Eval on GSM8K + MATH ────────────────────────────
BEST_MODEL="$MODEL_OUT/best_model"
if [ ! -d "$BEST_MODEL" ]; then
    BEST_MODEL="$MODEL_OUT"
fi

echo ""
echo "=== Step 2: Evaluate on GSM8K + MATH ==="
cd "$EVAL_DIR"

for task in gsm8k math; do
    echo ""
    echo "--- Eval: $task (zero-shot) ---"
    OUTPUT_DIR="${BEST_MODEL}/math_eval/test_cot-meta-math_zero-shot"

    python3 -u math_eval.py \
        --model_name_or_path "$BEST_MODEL" \
        --data_name "$task" \
        --output_dir "$OUTPUT_DIR" \
        --split test \
        --prompt_type cot-meta-math \
        --num_test_sample -1 \
        --seed 0 \
        --temperature 0 \
        --n_sampling 1 \
        --top_p 1 \
        --start 0 \
        --end -1 \
        --use_vllm \
        --save_outputs \
        --num_shots 0 \
        --dtype auto \
    || echo "[WARN] $task eval failed"
done

# ── Print results ────────────────────────────────────────────
echo ""
echo "================================================"
echo "  Results: 1K LEMMA Original (6800 entries), Full FT"
echo "================================================"
for task in gsm8k math; do
    METRICS=$(find "$BEST_MODEL/math_eval" -path "*${task}*" -name "*metrics.json" 2>/dev/null | head -1)
    if [ -n "$METRICS" ] && [ -f "$METRICS" ]; then
        python3 -c "import json; d=json.load(open('$METRICS')); print(f'  $task: {d[\"acc\"]}% ({d[\"num_samples\"]} samples)')"
    else
        echo "  $task: no results"
    fi
done
echo ""
echo "  Ended: $(date)"
echo "================================================"
