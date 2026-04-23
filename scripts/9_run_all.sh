#!/bin/bash
set -e

export TOKENIZERS_PARALLELISM=false
CONDA_BASE="/common/users/sl2148/anaconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate fact_yang

cd /common/users/sl2148/Public/yang_ouyang/projects/LostInTheSecond

echo "Python: $(which python3)"
echo "vllm: $(python3 -c 'import vllm; print(vllm.__version__)')"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

echo "============================================"
echo " Phase 0: Sample 8 CoT per MATH-500 question"
echo "   Model: Qwen2.5-3B-Instruct  GPUs=$GPUS"
echo "============================================"

python3 scripts/9_0_math500_sample_multi_cot.py \
    --n-samples 8 \
    --gpus "$GPUS"

echo ""
echo "============================================"
echo " Phase 1: GPT first-error locator + bucketing"
echo "============================================"

python3 scripts/9_1_find_first_error_and_bucket.py \
    --max-workers 8

echo ""
echo "============================================"
echo " Phase 2: Bad-prefix natural recovery (32 cont.)"
echo "   GPUs=$GPUS"
echo "============================================"

python3 scripts/9_2_bad_prefix_natural_recovery.py \
    --n-continuations 32 \
    --gpus "$GPUS"

echo ""
echo "============================================"
echo " Phase 3: Minimal repair continuation (32 cont.)"
echo "   GPUs=$GPUS"
echo "============================================"

python3 scripts/9_3_minimal_repair_continuation.py \
    --n-continuations 32 \
    --gpus "$GPUS"

echo ""
echo "============================================"
echo " Phase 4: Compare bad vs repair (per-sample + per-question)"
echo "============================================"

python3 scripts/9_4_compare_bad_vs_repair.py
python3 scripts/9_4_compare_bad_vs_repair.py --per-question

echo ""
echo "============================================"
echo " Phase 5: Fine-grained relative position analysis"
echo "============================================"

python3 scripts/9_5_fine_grained_relpos.py

echo ""
echo "============================================"
echo " ALL DONE -- MATH-500 Bad-Prefix Recovery"
echo "============================================"
echo "Figures: figures/math500_bad_prefix_recovery/"
echo "Results: results/math500_3b_multi_sample/"
