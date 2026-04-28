#!/bin/bash
#SBATCH --job-name=fft-lemma
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/jet/home/swang47/yang/projects/LostInTheSecond/logs/full_ft-%j.out
#SBATCH --error=/jet/home/swang47/yang/projects/LostInTheSecond/logs/full_ft-%j.err

set -euo pipefail

module load cuda/12.6.1
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

export HF_TOKEN=$(cat /jet/home/swang47/.cache/huggingface/token)
export PYTHONUNBUFFERED=1

cd /jet/home/swang47/yang/projects/LostInTheSecond

DATASET="${DATASET:-./artifacts_real/lemma_sft_wait_recompute_all.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/ocean/projects/cis250050p/swang47/yang/lostinsecond-models/full_ft_llama3_8b_lemma_wait_recompute}"

echo "================================================"
echo "  Full Fine-tune LLaMA-3-8B on LEMMA data"
echo "  Node: $(hostname)"
echo "  Dataset: $DATASET"
echo "  Output:  $OUTPUT_DIR"
echo "  Started: $(date)"
echo "================================================"

nvidia-smi

python3 scripts/data_prep/15_finetune_lemma.py \
    --method full \
    --model-id meta-llama/Meta-Llama-3-8B \
    --dataset-path "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --num-train-epochs 3 \
    --learning-rate 1e-5 \
    --warmup-ratio 0.03 \
    --weight-decay 0.01 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --max-length 2048 \
    --logging-steps 5 \
    --save-steps 50 \
    --eval-steps 50 \
    --optim adafactor

echo "Done: $(date)"
