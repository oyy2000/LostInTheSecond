#!/bin/bash
#SBATCH --job-name=prefill-s3
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --account=cis250050p
#SBATCH --output=/jet/home/swang47/yang/projects/LostInTheSecond/logs/prefill_s3_%j.out
#SBATCH --error=/jet/home/swang47/yang/projects/LostInTheSecond/logs/prefill_s3_%j.err

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
echo "  Step 3: Prefill + Convert"
echo "  Started: $(date)"
echo "======================================================"

# Prefill fix_step3
echo "--- Prefill fix_step3 ---"
python3 -u scripts/data_prep/13_prefill_llama3.py \
    --in-file artifacts_real/full/lemma_ds2_fix_step3_gpt.json \
    --out-file artifacts_real/full/lemma_ds2_fix_step3_gpt_prefill.json \
    --use-vllm --tensor-parallel 2 --dtype float16

# Convert to SFT format (correct only)
echo ""
echo "--- Convert fix_step3 (correct only) ---"
python3 -u scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file artifacts_real/full/lemma_ds2_fix_step3_gpt_prefill.json \
    --out-file artifacts_real/full/lemma_sft_fix_step3.json \
    --require-correct

# Also convert all (wait_recompute style)
echo ""
echo "--- Convert wait_step3 ---"
python3 -u scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file artifacts_real/full/lemma_ds2_wait_step3_gpt.json \
    --out-file artifacts_real/full/lemma_sft_wait_step3.json \
    --no-require-correct

echo ""
echo "======================================================"
echo "  Completed: $(date)"
echo "======================================================"
