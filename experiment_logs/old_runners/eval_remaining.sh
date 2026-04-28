#!/bin/bash

cd /common/users/sl2148/Public/yang_ouyang/projects/LostInTheSecond

PYTHON_BIN="/common/home/sl2148/anaconda3/envs/sft/bin/python"
EVAL_ROOT="artifacts_local/lora_sweep/_gsm8k_vllm_eval"
MERGED_BASE="artifacts_local/lora_sweep"
VLLM_ARGS="dtype=float16,gpu_memory_utilization=0.9,max_model_len=2048,max_num_seqs=16,enforce_eager=True"

echo "=== Step 1: Kill ALL lm_eval and vllm processes ==="
pkill -9 -f "lm_eval" 2>/dev/null || true
pkill -9 -f "vllm" 2>/dev/null || true
echo "Waiting 30 seconds for GPU memory to free..."
sleep 30

echo "=== Step 2: Check GPU status ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

# Find first free GPU
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 100 {print $1; exit}')
echo "Selected GPU: $GPU"

if [ -z "$GPU" ]; then
    echo "No free GPU found. Waiting 60 more seconds..."
    sleep 60
    GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 100 {print $1; exit}')
    echo "Retry selected GPU: $GPU"
fi

if [ -z "$GPU" ]; then
    echo "ERROR: No free GPU available"
    exit 1
fi

# These are the adapters that need evaluation (missing or bad results)
ADAPTERS="v3_lr1e5_r16_attn v3_lr1e5_r4_a1x v3_lr1e6_r4_attn v3_lr5e6_r4_a1x v3_lr5e6_r4_attn"

echo ""
echo "=== Step 3: Evaluate $ADAPTERS ==="
echo ""

for ADAPTER_NAME in $ADAPTERS; do
    MERGED_PATH="${MERGED_BASE}/${ADAPTER_NAME}/merged_model"
    if [ ! -f "${MERGED_PATH}/config.json" ]; then
        echo "SKIP: ${ADAPTER_NAME} (no merged model)"
        continue
    fi

    # Clear the old eval directory
    rm -rf "${EVAL_ROOT}/${ADAPTER_NAME}" 2>/dev/null || true

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
        --apply_chat_template

    EVAL_EXIT=$?
    echo "Exit code: $EVAL_EXIT"
    echo ""

    # Quick check for result
    RESULT_FILE=$(find "${EVAL_ROOT}/${ADAPTER_NAME}" -name "results_*.json" 2>/dev/null | head -1)
    if [ -n "$RESULT_FILE" ]; then
        echo "Result found: $RESULT_FILE"
        $PYTHON_BIN -c "
import json
obj = json.loads(open('$RESULT_FILE').read())
task = (obj.get('results') or {}).get('gsm8k_cot_zeroshot_unified', {})
for key, val in task.items():
    if 'exact_match' in key and 'stderr' not in key:
        print(f'  {key}: {val}')
"
    else
        echo "WARNING: No result file found!"
    fi
    echo ""
done

echo "=== Step 4: Rebuild all_results.json ==="
$PYTHON_BIN -c "
import json
from pathlib import Path

eval_root = Path('$EVAL_ROOT')
all_results = json.load(open(eval_root / 'all_results.json'))

# Update results from evaluation directories
for entry in all_results:
    name = entry['name']
    if name == 'base':
        continue
    eval_dir = eval_root / name
    for p in sorted(eval_dir.rglob('results_*.json')):
        obj = json.loads(p.read_text())
        task = (obj.get('results') or {}).get('gsm8k_cot_zeroshot_unified', {})
        for key, val in task.items():
            if 'exact_match' in key and 'stderr' not in key:
                try:
                    entry['gsm8k_em'] = float(val)
                except:
                    pass
                break
        break

# Check for any adapters not in all_results
existing_names = set(e['name'] for e in all_results)
for d in sorted(eval_root.iterdir()):
    if not d.is_dir() or d.name in existing_names or d.name.startswith('_'):
        continue
    for p in d.rglob('results_*.json'):
        obj = json.loads(p.read_text())
        task = (obj.get('results') or {}).get('gsm8k_cot_zeroshot_unified', {})
        for key, val in task.items():
            if 'exact_match' in key and 'stderr' not in key:
                try:
                    all_results.append({'name': d.name, 'gsm8k_em': float(val)})
                except:
                    pass
                break
        break

json.dump(all_results, open(eval_root / 'all_results.json', 'w'), indent=2, default=str)
print('Updated all_results.json')

# Print summary
base_em = next((e['gsm8k_em'] for e in all_results if e['name'] == 'base'), 0)
print(f'\nBase: {base_em:.4f}')
results = [(e['name'], e.get('gsm8k_em')) for e in all_results if e['name'] != 'base']
results.sort(key=lambda x: -(x[1] or -1))
for name, em in results:
    if em is not None:
        delta = em - base_em
        print(f'{name}: EM={em:.4f}  delta={delta:+.4f}')
    else:
        print(f'{name}: MISSING')
"

echo ""
echo "=== All done ==="
echo "Time: $(date)"
