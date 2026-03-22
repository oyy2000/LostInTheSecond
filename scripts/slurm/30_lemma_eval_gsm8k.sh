#!/bin/bash
#SBATCH --job-name=lemma-gsm8k
#SBATCH --partition=GPU-shared
#SBATCH --account=cis250050p
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs/lemma_gsm8k_%j.out
#SBATCH --error=/ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs/lemma_gsm8k_%j.out

set -euo pipefail

module load cuda/12.6.1
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

export HF_TOKEN=$(cat /jet/home/swang47/.cache/huggingface/token)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="${1:?Usage: sbatch ... script.sh MODEL_NAME MODEL_PATH}"
MODEL_PATH="${2:?Usage: sbatch ... script.sh MODEL_NAME MODEL_PATH}"

EVAL_DIR=/jet/home/swang47/yang/projects/LEMMA/evaluation
OUTPUT_DIR="${MODEL_PATH}/math_eval/test_cot-meta-math_zero-shot"

echo "======================================================"
echo "  LEMMA GSM8K Eval: ${MODEL_NAME}"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  Model: ${MODEL_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Started: $(date)"
echo "======================================================"

cd "${EVAL_DIR}"

echo ""
echo "--- Eval: gsm8k (zero-shot, cot-meta-math) ---"
python3 -u math_eval.py \
    --data_names gsm8k \
    --model_name_or_path "${MODEL_PATH}" \
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
    --num_shots 0 \
    --dtype float16

echo ""
echo "======================================================"
echo "  Result: ${MODEL_NAME}"
echo "======================================================"
METRICS=$(find "${OUTPUT_DIR}" -path "*gsm8k*" -name "*metrics*" 2>/dev/null | head -1)
if [ -n "$METRICS" ] && [ -f "$METRICS" ]; then
    cat "$METRICS"
else
    echo "  No metrics file found"
fi
echo ""
echo "  Ended: $(date)"
echo "======================================================"
