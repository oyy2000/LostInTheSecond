#!/bin/bash
#SBATCH --job-name=rej-sample
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --account=cis250050p
#SBATCH --output=/jet/home/swang47/yang/projects/LostInTheSecond/logs/rej_sample_%j.out
#SBATCH --error=/jet/home/swang47/yang/projects/LostInTheSecond/logs/rej_sample_%j.err

set -euo pipefail

module load cuda/12.6.1
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

export HF_TOKEN=$(cat /jet/home/swang47/.cache/huggingface/token)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PROJECT=/jet/home/swang47/yang/projects/LostInTheSecond
cd $PROJECT

echo "======================================================"
echo "  Rejection Sampling: LLaMA-3-8B-Instruct"
echo "  Node: $(hostname)"
echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  Started: $(date)"
echo "======================================================"

python3 -u scripts/data_prep/18_generate_multi_sample.py \
    --in-file artifacts_real/full/lemma_llama3_generations.json \
    --out-file artifacts_real/full/lemma_sft_rejection_sampled.json \
    --out-raw artifacts_real/full/lemma_rejection_raw.json \
    --model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --n-samples 8 \
    --temperature 0.7 \
    --top-p 0.95 \
    --tensor-parallel 2 \
    --dtype float16 \
    --only-previously-wrong

echo ""
echo "======================================================"
echo "  Rejection Sampling Complete"
echo "  Ended: $(date)"
echo "======================================================"
