#!/bin/bash
set -e

export TOKENIZERS_PARALLELISM=false
CONDA_BASE="/common/users/sl2148/anaconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate rl_steer_yang

cd /common/users/sl2148/Public/yang_ouyang/projects/LostInTheSecond

echo "Python: $(which python3)"
echo "vllm: $(python3 -c 'import vllm; print(vllm.__version__)')"

# Clean up any leftover artifacts
rm -rf results/gsm8k_3b_multi_sample/late_rollback/_shards
rm -f results/gsm8k_3b_multi_sample/late_rollback/drafts.jsonl
rm -rf results/gsm8k_3b_multi_sample/full_sc/_shards

echo "============================================"
echo " Phase 1: Late Rollback + Suffix Vote"
echo "   alpha=0.6,0.7,0.8  K=8  GPUs=2,3,4,5,6,7"
echo "============================================"

python3 scripts/7_1_late_rollback_suffix_vote.py \
    --alpha 0.6,0.7,0.8 \
    --K 8 \
    --gpus 2,3,4,5,6,7

echo ""
echo "============================================"
echo " Phase 2: Full Self-Consistency (K=8)"
echo "   GPUs=2,3,4,5,6,7"
echo "============================================"

python3 scripts/7_2_full_self_consistency.py \
    --K 8 \
    --gpus 2,3,4,5,6,7

echo ""
echo "============================================"
echo " Phase 3: Analysis + Figures"
echo "============================================"

python3 scripts/7_3_compare_late_rollback_vs_sc.py

echo ""
echo "============================================"
echo " ALL DONE"
echo "============================================"
