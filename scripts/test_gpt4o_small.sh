#!/bin/bash
set -e

source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

cd /jet/home/swang47/yang/projects/LostInTheSecond

export OPENAI_API_KEY="sk-sNqUCd2XmGHrzXoZDCTmLa7iHdi9L5paDJhTpyNfwkHW8Hc8"
export OPENAI_BASE_URL="https://yinli.one/v1"

echo "=== Small batch test: GPT-4o Step 2 Fix (3 samples) ==="
python scripts/data_prep/12_gpt_fix_step2_lemma.py \
    --in-file artifacts_real/full/lemma_llama3_generations.json \
    --out-fix /tmp/test_fix_step2.json \
    --out-wait /tmp/test_wait_step2.json \
    --audit-json /tmp/test_audit_step2.json \
    --model gpt-4o \
    --temperature 0.0 \
    --only-incorrect \
    --limit 3

echo ""
echo "=== Test results ==="
python -c "
import json
audit = json.load(open('/tmp/test_audit_step2.json'))
for a in audit:
    print(f'doc_id={a[\"doc_id\"]}, judged_correct={a[\"judged_correct\"]}')
    if not a['judged_correct']:
        print(f'  orig: {a[\"step2_orig\"][:100]}')
        print(f'  fix:  {a[\"step2_corrected\"][:100]}')
    print()
"
