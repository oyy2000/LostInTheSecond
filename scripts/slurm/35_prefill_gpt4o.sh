#!/bin/bash
#SBATCH --job-name=prefill-gpt4o
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --account=cis250050p
#SBATCH --output=/jet/home/swang47/yang/projects/LostInTheSecond/logs/prefill-gpt4o-%j.out
#SBATCH --error=/jet/home/swang47/yang/projects/LostInTheSecond/logs/prefill-gpt4o-%j.err

set -euo pipefail

module load cuda/12.6.1
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

export HF_TOKEN=$(cat /jet/home/swang47/.cache/huggingface/token)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd /jet/home/swang47/yang/projects/LostInTheSecond
IN_DIR="./artifacts_real/full_gpt4o"
TP=2

echo "================================================"
echo "  Prefill + Convert (GPT-4o data, vLLM, tp=$TP)"
echo "  Node: $(hostname)"
echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "  Started: $(date)"
echo "================================================"
nvidia-smi

# ── Phase 1: Prefill Step 2 fix variant ──
echo ""
echo "=== Prefill Step 2 fix variant ==="
python3 -u scripts/data_prep/13_prefill_llama3.py \
    --in-file "$IN_DIR/lemma_ds2_fix_step2_gpt4o.json" \
    --out-file "$IN_DIR/lemma_ds2_fix_step2_gpt4o_prefill.json" \
    --keep-steps 2 \
    --max-attempts 5 \
    --use-vllm \
    --tensor-parallel $TP \
    --dtype float16

# ── Phase 2: Prefill Step 2 wait variant ──
echo ""
echo "=== Prefill Step 2 wait_recompute variant ==="
python3 -u scripts/data_prep/13_prefill_llama3.py \
    --in-file "$IN_DIR/lemma_ds2_wait_recompute_gpt4o.json" \
    --out-file "$IN_DIR/lemma_ds2_wait_recompute_gpt4o_prefill.json" \
    --keep-steps 0 \
    --max-attempts 5 \
    --use-vllm \
    --tensor-parallel $TP \
    --dtype float16

# ── Phase 3: Prefill Step 3 fix variant ──
echo ""
echo "=== Prefill Step 3 fix variant ==="
python3 -u scripts/data_prep/13_prefill_llama3.py \
    --in-file "$IN_DIR/lemma_ds2_fix_step3_gpt4o.json" \
    --out-file "$IN_DIR/lemma_ds2_fix_step3_gpt4o_prefill.json" \
    --keep-steps 3 \
    --max-attempts 5 \
    --use-vllm \
    --tensor-parallel $TP \
    --dtype float16

# ── Phase 4: Prefill Step 3 wait variant ──
echo ""
echo "=== Prefill Step 3 wait variant ==="
python3 -u scripts/data_prep/13_prefill_llama3.py \
    --in-file "$IN_DIR/lemma_ds2_wait_step3_gpt4o.json" \
    --out-file "$IN_DIR/lemma_ds2_wait_step3_gpt4o_prefill.json" \
    --keep-steps 0 \
    --max-attempts 5 \
    --use-vllm \
    --tensor-parallel $TP \
    --dtype float16

# ── Phase 5: Convert to SFT format ──
echo ""
echo "=== Convert to SFT format ==="

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$IN_DIR/lemma_ds2_fix_step2_gpt4o_prefill.json" \
    --out-file "$IN_DIR/lemma_sft_fix_step2.json"

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$IN_DIR/lemma_ds2_wait_recompute_gpt4o_prefill.json" \
    --out-file "$IN_DIR/lemma_sft_wait_recompute.json"

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$IN_DIR/lemma_ds2_wait_recompute_gpt4o_prefill.json" \
    --out-file "$IN_DIR/lemma_sft_wait_recompute_all.json" \
    --no-require-correct

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$IN_DIR/lemma_ds2_fix_step3_gpt4o_prefill.json" \
    --out-file "$IN_DIR/lemma_sft_fix_step3.json"

python3 scripts/data_prep/14_convert_to_lemma_format.py \
    --in-file "$IN_DIR/lemma_ds2_wait_step3_gpt4o_prefill.json" \
    --out-file "$IN_DIR/lemma_sft_wait_step3.json"

# ── Phase 6: Build combined SFT dataset ──
echo ""
echo "=== Build combined SFT dataset ==="
python3 scripts/data_prep/21_build_combined_sft.py \
    --greedy-gen artifacts_real/full/lemma_llama3_generations.json \
    --prefill-step2 "$IN_DIR/lemma_sft_fix_step2.json" \
    --prefill-step3 "$IN_DIR/lemma_sft_fix_step3.json" \
    --rejection-sampled artifacts_real/full/lemma_sft_rejection_sampled.json \
    --out-file "$IN_DIR/lemma_sft_combined_gpt4o.json"

echo ""
echo "================================================"
echo "  ALL PHASES COMPLETE!"
echo "  Ended: $(date)"
echo "================================================"
ls -lh "$IN_DIR"/lemma_sft_*.json
