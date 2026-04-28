#!/bin/bash
#SBATCH --job-name=fp-compare
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --account=cis250050p
#SBATCH --output=/jet/home/swang47/yang/projects/LostInTheSecond/logs/fp_compare_%j.out
#SBATCH --error=/jet/home/swang47/yang/projects/LostInTheSecond/logs/fp_compare_%j.err

set -euo pipefail

module load cuda/12.6.1
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

export HF_TOKEN=$(cat /jet/home/swang47/.cache/huggingface/token)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /jet/home/swang47/yang/projects/LostInTheSecond

echo "======================================================"
echo "  Full Prefill Pipeline Comparison"
echo "  Node: $(hostname)"
echo "  GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  Started: $(date)"
echo "======================================================"

python3 -u scripts/eval/31_train_and_eval_full_prefill.py --gpus 0,1

echo ""
echo "======================================================"
echo "  Completed: $(date)"
echo "======================================================"
