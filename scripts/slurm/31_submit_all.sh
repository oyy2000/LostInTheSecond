#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/31_single_train.sh"
DATA_ROOT="/jet/home/swang47/yang/projects/LostInTheSecond/artifacts_real/full"
ALL_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

echo "Submitting 4 training jobs (1x H100 each)..."
echo ""

# LoRA fix_step2 (498 samples)
JOB1=$(sbatch --export=ALL --parsable "$TRAIN_SCRIPT" \
    lora_fix498 lora "$DATA_ROOT/lemma_sft_fix_step2.json" \
    5e-6 5 8 0.03 0.0 \
    --lora-r 4 --lora-alpha 4 --lora-dropout 0.05 \
    --target-modules "$ALL_MODULES" --eval-steps 9999 --save-steps 9999)
echo "  lora_fix498:  job $JOB1"

# LoRA wait_recompute_all (2224 samples)
JOB2=$(sbatch --export=ALL --parsable "$TRAIN_SCRIPT" \
    lora_wr2224 lora "$DATA_ROOT/lemma_sft_wait_recompute_all.json" \
    5e-6 3 16 0.03 0.0 \
    --lora-r 4 --lora-alpha 4 --lora-dropout 0.05 \
    --target-modules "$ALL_MODULES" --eval-steps 9999 --save-steps 9999)
echo "  lora_wr2224:  job $JOB2"

# Full FT fix_step2 (498 samples)
JOB3=$(sbatch --export=ALL --parsable "$TRAIN_SCRIPT" \
    ft_fix498 full "$DATA_ROOT/lemma_sft_fix_step2.json" \
    1e-6 5 8 0.1 0.05 \
    --optim adafactor --eval-steps 7 --save-steps 7)
echo "  ft_fix498:    job $JOB3"

# Full FT wait_recompute_all (2224 samples)
JOB4=$(sbatch --export=ALL --parsable "$TRAIN_SCRIPT" \
    ft_wr2224 full "$DATA_ROOT/lemma_sft_wait_recompute_all.json" \
    1e-6 3 16 0.1 0.05 \
    --optim adafactor --eval-steps 7 --save-steps 7)
echo "  ft_wr2224:    job $JOB4"

echo ""
echo "All 4 jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "After training completes, submit eval jobs with:"
echo "  bash scripts/slurm/31_submit_evals.sh"
