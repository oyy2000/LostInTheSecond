#!/bin/bash

cd /common/users/sl2148/Public/yang_ouyang/projects/LostInTheSecond

PYTHON_BIN="/common/home/sl2148/anaconda3/envs/sft/bin/python"
EVAL_ROOT="artifacts_local/lora_sweep/_gsm8k_vllm_eval"
MERGED_BASE="artifacts_local/lora_sweep"
VLLM_ARGS="dtype=float16,gpu_memory_utilization=0.9,max_model_len=2048,max_num_seqs=16,enforce_eager=True"
GPU=0

echo "=== Checking for running lm_eval processes ==="
ps -ef | grep "lm_eval" | grep -v grep || echo "No lm_eval processes running"
echo ""

echo "=== Checking GPU usage ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo ""

echo "=== Checking existing results ==="
$PYTHON_BIN -c "
import json
data = json.load(open('$EVAL_ROOT/all_results.json'))
done = []
missing = []
for entry in data:
    name = entry['name']
    em = entry.get('gsm8k_em')
    if em is not None:
        done.append(f'{name}: EM={em:.4f}')
    else:
        missing.append(name)
print(f'Done ({len(done)}):')
for d in done:
    print(f'  {d}')
print(f'Missing ({len(missing)}):')
for m in missing:
    print(f'  {m}')
"
echo ""

# Check which adapters still need evaluation
NEED_EVAL=$($PYTHON_BIN -c "
import json
data = json.load(open('$EVAL_ROOT/all_results.json'))
missing = [e['name'] for e in data if e.get('gsm8k_em') is None]
# Also check for adapters not in the results at all
import os
all_adapters = set()
for d in os.listdir('$MERGED_BASE'):
    mp = os.path.join('$MERGED_BASE', d, 'merged_model', 'config.json')
    if os.path.exists(mp):
        all_adapters.add(d)
in_results = set(e['name'] for e in data)
extra = all_adapters - in_results
missing.extend(sorted(extra))
print(' '.join(missing))
")

if [ -z "$NEED_EVAL" ]; then
    echo "All adapters evaluated!"
    exit 0
fi

echo "=== Need evaluation: $NEED_EVAL ==="
echo ""

# Kill any lingering processes first
pkill -f "lm_eval.*vllm" 2>/dev/null || true
sleep 10

for ADAPTER_NAME in $NEED_EVAL; do
    MERGED_PATH="${MERGED_BASE}/${ADAPTER_NAME}/merged_model"
    if [ ! -f "${MERGED_PATH}/config.json" ]; then
        echo "SKIP: ${ADAPTER_NAME} (no merged model)"
        continue
    fi

    # Check if results already exist for this adapter (maybe completed while we were running)
    HAS_RESULT=$($PYTHON_BIN -c "
import json, sys
from pathlib import Path
for p in sorted(Path('$EVAL_ROOT/$ADAPTER_NAME').rglob('results_*.json')):
    obj = json.loads(p.read_text())
    task_block = (obj.get('results') or {}).get('gsm8k_cot_zeroshot_unified', {})
    for key, val in task_block.items():
        if 'exact_match' in key and 'stderr' not in key:
            print(f'{val}')
            sys.exit(0)
print('none')
" 2>/dev/null)

    if [ "$HAS_RESULT" != "none" ] && [ -n "$HAS_RESULT" ]; then
        echo "SKIP: ${ADAPTER_NAME} (already has result: EM=${HAS_RESULT})"
        continue
    fi

    echo "================================================================"
    echo "EVALUATING: ${ADAPTER_NAME} on GPU ${GPU}"
    echo "Time: $(date)"
    echo "================================================================"

    CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false \
    $PYTHON_BIN -m lm_eval \
        --model vllm \
        --model_args "pretrained=${MERGED_PATH},${VLLM_ARGS}" \
        --tasks gsm8k_cot_zeroshot_unified \
        --batch_size auto \
        --gen_kwargs "max_gen_toks=2048,temperature=0,do_sample=False" \
        --output_path "${EVAL_ROOT}/${ADAPTER_NAME}" \
        --log_samples \
        --apply_chat_template \
        2>&1 | tee "${EVAL_ROOT}/${ADAPTER_NAME}/eval_direct.log"

    EVAL_EXIT=$?
    echo "Exit code: $EVAL_EXIT"
    echo ""
done

echo "=== All evaluations done ==="
echo "Time: $(date)"

# Print final results
echo ""
echo "=== Final Results ==="
$PYTHON_BIN -c "
import json
from pathlib import Path
eval_root = Path('$EVAL_ROOT')
results = {}
# Load from all_results.json
for entry in json.load(open(eval_root / 'all_results.json')):
    if entry.get('gsm8k_em') is not None:
        results[entry['name']] = entry['gsm8k_em']
# Also scan for new results
for d in sorted(eval_root.iterdir()):
    if not d.is_dir() or d.name in results:
        continue
    for p in d.rglob('results_*.json'):
        obj = json.loads(p.read_text())
        task = (obj.get('results') or {}).get('gsm8k_cot_zeroshot_unified', {})
        for key, val in task.items():
            if 'exact_match' in key and 'stderr' not in key:
                try:
                    results[d.name] = float(val)
                except:
                    pass
                break

base_em = results.get('base', 0)
print(f'Base: {base_em:.4f}')
print(f'')
sorted_results = sorted(results.items(), key=lambda x: -x[1] if x[0] != 'base' else 0)
for name, em in sorted_results:
    if name == 'base':
        continue
    delta = em - base_em
    print(f'{name}: EM={em:.4f}  delta={delta:+.4f}')
"
