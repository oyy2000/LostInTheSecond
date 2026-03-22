#!/bin/bash
#SBATCH --job-name=eval-lits
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=/jet/home/swang47/yang/projects/LostInTheSecond/logs/eval-%j.out
#SBATCH --error=/jet/home/swang47/yang/projects/LostInTheSecond/logs/eval-%j.err

set -euo pipefail

module load cuda/12.6.1
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

export HF_TOKEN=$(cat /jet/home/swang47/.cache/huggingface/token)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PROJ=/jet/home/swang47/yang/projects/LostInTheSecond
EVAL_DIR=/jet/home/swang47/yang/projects/LEMMA/evaluation
LORA_ADAPTER=/ocean/projects/cis250050p/swang47/yang/lostinsecond-models/lora_llama3_8b_lemma_wait_recompute/final_adapter
LORA_MERGED=/ocean/projects/cis250050p/swang47/yang/lostinsecond-models/lora_llama3_8b_lemma_wait_recompute/merged_model
FULL_FT=/ocean/projects/cis250050p/swang47/yang/lostinsecond-models/full_ft_llama3_8b_lemma_wait_recompute/best_model

echo "================================================"
echo "  Evaluate LostInTheSecond fine-tuned models"
echo "  Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Python: $(which python3)"
echo "  Started: $(date)"
echo "================================================"
nvidia-smi

# ── Step 1: Merge LoRA adapter ───────────────────────────────
if [ ! -f "$LORA_MERGED/model.safetensors.index.json" ] || [ -z "$(find "$LORA_MERGED" -name '*.safetensors' 2>/dev/null)" ]; then
    echo ""
    echo "=== Step 1: Merge LoRA adapter ==="
    rm -rf "$LORA_MERGED"
    cd "$PROJ"
    python3 scripts/eval/merge_lora_and_eval.py \
        --base-model meta-llama/Meta-Llama-3-8B \
        --adapter-path "$LORA_ADAPTER" \
        --output-dir "$LORA_MERGED"
    echo "Merge done. Files:"
    ls -lh "$LORA_MERGED"/*.safetensors 2>/dev/null || echo "WARNING: no safetensors found!"
else
    echo "=== Step 1: LoRA merged model exists, skipping ==="
fi

# ── Step 2: Evaluate ─────────────────────────────────────────
cd "$EVAL_DIR"

run_eval() {
    local MODEL_PATH="$1"
    local DATA_NAME="$2"
    local NUM_SHOTS="$3"
    local LABEL="$4"

    local SHOT_LABEL="zero-shot"
    if [ "$NUM_SHOTS" -gt 0 ]; then
        SHOT_LABEL="${NUM_SHOTS}-shot"
    fi

    local OUTPUT_DIR="${MODEL_PATH}/math_eval/test_cot-meta-math_${SHOT_LABEL}"

    echo ""
    echo "--- ${LABEL}: ${DATA_NAME} (${SHOT_LABEL}) ---"

    python3 -u math_eval.py \
        --model_name_or_path "${MODEL_PATH}" \
        --data_name "${DATA_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
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
        --num_shots "${NUM_SHOTS}" \
        --dtype half \
    || echo "[WARN] ${DATA_NAME} eval failed for ${LABEL}"
}

MODELS=("$LORA_MERGED" "$FULL_FT")
NAMES=("LoRA-merged" "Full-FT")

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    NAME="${NAMES[$i]}"

    echo ""
    echo "========================================"
    echo "  Evaluating: $NAME"
    echo "  Path: $MODEL"
    echo "========================================"

    # In-domain (zero-shot)
    run_eval "$MODEL" "gsm8k" 0 "$NAME"
    run_eval "$MODEL" "math"  0 "$NAME"
done

# ── Print results summary ────────────────────────────────────
echo ""
echo "================================================"
echo "  Results Summary"
echo "================================================"
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    NAME="${NAMES[$i]}"
    echo ""
    echo "--- $NAME ---"
    for task in gsm8k math; do
        METRICS=$(find "$MODEL/math_eval" -path "*${task}*" -name "*metrics.json" 2>/dev/null | head -1)
        if [ -n "$METRICS" ] && [ -f "$METRICS" ]; then
            ACC=$(python3 -c "import json; d=json.load(open('$METRICS')); print(f'{task}: {d.get(\"acc\", \"N/A\")}%')" 2>/dev/null || echo "$task: parse error")
            echo "  $ACC"
        else
            echo "  $task: no results"
        fi
    done
done

echo ""
echo "  Ended: $(date)"
echo "================================================"
