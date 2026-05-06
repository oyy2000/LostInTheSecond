#!/usr/bin/env bash
set -euo pipefail

# CommonsenseQA bad-prefix recovery pipeline
# Usage: bash scripts/10_run_all.sh

cd "$(dirname "$0")/.."
source /common/users/sl2148/anaconda3/etc/profile.d/conda.sh
conda activate fact_yang
export TOKENIZERS_PARALLELISM=false

GPUS="0,1,2,3,4,5,6,7"

echo "=== Phase 0: Sample 8 CoT per CSQA question ==="
python3 scripts/10_0_csqa_sample_multi_cot.py --n-samples 8 --gpus $GPUS

echo ""
echo "=== Phase 1: GPT first-error locator ==="
python3 scripts/10_1_csqa_find_first_error.py --max-workers 8

echo ""
echo "=== Phase 2: Bad-prefix natural recovery ==="
python3 scripts/10_2_csqa_bad_prefix_recovery.py --n-continuations 32 --gpus $GPUS --gpu-memory-utilization 0.95

echo ""
echo "=== Phase 3: Minimal repair continuation ==="
python3 scripts/10_3_csqa_minimal_repair.py --n-continuations 32 --gpus $GPUS --gpu-memory-utilization 0.95

echo ""
echo "=== Phase 4: Compare bad vs repair ==="
python3 scripts/10_4_csqa_compare.py

echo ""
echo "=== Phase 5: Fine-grained relative position ==="
python3 scripts/10_5_csqa_fine_relpos.py

echo ""
echo "=== All done ==="
