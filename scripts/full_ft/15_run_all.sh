#!/bin/bash
set -euo pipefail

PYTHON=/mnt/beegfs/youyang7/.conda/envs/fact/bin/python
SCRIPT=/home/youyang7/projects/LostInTheSecond/scripts/full_ft/13_full_finetune.py
EVAL_SCRIPT=/home/youyang7/projects/LostInTheSecond/scripts/full_ft/14_full_ft_sweep_and_eval.py
DATASET=/mnt/beegfs/youyang7/projects/LostInSecond/artifacts/samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json
SWEEP=/mnt/beegfs/youyang7/projects/LostInSecond/artifacts/full_ft_sweep
EVAL_DIR=${SWEEP}/_gsm8k_vllm_eval
MODEL=Qwen/Qwen2.5-3B-Instruct
HARNESS_DIR=/home/youyang7/projects/LostInTheSecond/lm-evaluation-harness
TASK=gsm8k_cot_zeroshot_unified

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

VLLM_ARGS="pretrained=__MODEL__,dtype=bfloat16,gpu_memory_utilization=0.55,max_model_len=3072,max_num_seqs=16,enforce_eager=True"

echo "============================================================"
echo "  FULL FT SWEEP: TRAIN + GSM8K EVAL PIPELINE"
echo "  Started: $(date)"
echo "============================================================"

# ---------------------------------------------------------------
# PHASE 1: Train the 3 remaining experiments in parallel
# ---------------------------------------------------------------
train_one() {
    local name=$1 lr=$2 epochs=$3 wd=$4 warmup=$5 gpu=$6
    local outdir="${SWEEP}/${name}"

    if [ -f "${outdir}/best_model/config.json" ] && [ -f "${outdir}/sweep_metrics.json" ]; then
        echo "[SKIP] ${name} already complete"
        return 0
    fi

    rm -rf "${outdir}" 2>/dev/null
    mkdir -p "${outdir}"
    echo "[TRAIN] ${name} (lr=${lr}, ep=${epochs}, wd=${wd}, warmup=${warmup}) on GPU ${gpu}"

    CUDA_VISIBLE_DEVICES=${gpu} $PYTHON -u $SCRIPT \
        --model-id $MODEL \
        --dataset-path $DATASET \
        --output-dir "${outdir}" \
        --learning-rate ${lr} \
        --num-train-epochs ${epochs} \
        --weight-decay ${wd} \
        --warmup-ratio ${warmup} \
        --gradient-accumulation-steps 16 \
        --per-device-train-batch-size 1 \
        --logging-steps 1 --eval-steps 7 --save-steps 7 --save-total-limit 2 --seed 42 \
        > "${outdir}/training.log" 2>&1

    if [ -f "${outdir}/best_model/config.json" ]; then
        echo "[OK] ${name} training complete"
    else
        echo "[FAIL] ${name} training failed"
    fi
}

echo ""
echo "=== PHASE 1: Training 3 remaining experiments ==="
echo ""

train_one ft_lr1e5  1e-5  3 0.01 0.1 0 &
PID_LR1E5=$!

train_one ft_lr2e5  2e-5  3 0.01 0.1 1 &
PID_LR2E5=$!

train_one ft_lr5e5  5e-5  3 0.01 0.1 2 &
PID_LR5E5=$!

echo "Waiting for training jobs (PIDs: $PID_LR1E5, $PID_LR2E5, $PID_LR5E5)..."

TRAIN_FAIL=0
for pid in $PID_LR1E5 $PID_LR2E5 $PID_LR5E5; do
    wait $pid || TRAIN_FAIL=$((TRAIN_FAIL + 1))
done

echo ""
echo "=== Training summary ==="
TOTAL_COMPLETE=0
TOTAL_INCOMPLETE=0
for d in ${SWEEP}/ft_*/; do
    name=$(basename "$d")
    if [ -f "${d}best_model/config.json" ] && [ -f "${d}sweep_metrics.json" ]; then
        echo "  COMPLETE: $name"
        TOTAL_COMPLETE=$((TOTAL_COMPLETE + 1))
    else
        echo "  INCOMPLETE: $name"
        TOTAL_INCOMPLETE=$((TOTAL_INCOMPLETE + 1))
    fi
done
echo "  Total: ${TOTAL_COMPLETE} complete, ${TOTAL_INCOMPLETE} incomplete"
echo ""

if [ $TOTAL_INCOMPLETE -gt 0 ] && [ $TRAIN_FAIL -gt 0 ]; then
    echo "[WARN] Some training failed. Continuing with evaluation of completed models."
fi

# ---------------------------------------------------------------
# PHASE 2: GSM8K Evaluation (one model per GPU, strict isolation)
# ---------------------------------------------------------------
eval_one() {
    local model_path=$1 output_dir=$2 gpu=$3 name=$4

    mkdir -p "${output_dir}"
    local model_args
    model_args=$(echo "$VLLM_ARGS" | sed "s|__MODEL__|${model_path}|g")

    echo "[EVAL] ${name} on GPU ${gpu}..."
    cd "${HARNESS_DIR}"
    CUDA_VISIBLE_DEVICES=${gpu} $PYTHON -m lm_eval \
        --model vllm \
        --model_args "${model_args}" \
        --tasks ${TASK} \
        --batch_size auto \
        --gen_kwargs "max_gen_toks=2048,temperature=0,do_sample=False" \
        --output_path "${output_dir}" \
        --log_samples \
        --apply_chat_template \
        > "${output_dir}/eval.log" 2>&1

    local result_file
    result_file=$(find "${output_dir}" -name "results_*.json" -type f 2>/dev/null | sort | tail -1)
    if [ -n "$result_file" ]; then
        local em
        em=$($PYTHON -c "
import json, sys
obj = json.load(open('${result_file}'))
task = obj.get('results',{}).get('${TASK}',{})
for k in ['exact_match,flexible-extract','exact_match,strict-match','exact_match,none']:
    if k in task:
        print(f'{task[k]:.4f}')
        sys.exit(0)
print('N/A')
" 2>/dev/null || echo "N/A")
        echo "[OK] ${name}: GSM8K EM = ${em}"
    else
        echo "[FAIL] ${name}: no results file found"
    fi
}

set +e
echo "=== PHASE 2: GSM8K Evaluation ==="
echo ""

# Build list of models to evaluate
declare -a EVAL_NAMES=()
declare -a EVAL_PATHS=()

# Base model
EVAL_NAMES+=("base")
EVAL_PATHS+=("${MODEL}")

# All completed full FT models
for d in ${SWEEP}/ft_*/; do
    name=$(basename "$d")
    model_path="${d}best_model"
    if [ -f "${model_path}/config.json" ]; then
        EVAL_NAMES+=("${name}")
        EVAL_PATHS+=("${model_path}")
    fi
done

TOTAL_EVAL=${#EVAL_NAMES[@]}
echo "Models to evaluate: ${TOTAL_EVAL}"

# Check which already have valid results
declare -a TODO_NAMES=()
declare -a TODO_PATHS=()
declare -a DONE_NAMES=()

for i in $(seq 0 $((TOTAL_EVAL - 1))); do
    name=${EVAL_NAMES[$i]}
    eval_dir="${EVAL_DIR}/${name}"
    result_file=$(find "${eval_dir}" -name "results_*.json" -type f 2>/dev/null | sort | tail -1)
    if [ -n "$result_file" ]; then
        em=$($PYTHON -c "
import json, sys
obj = json.load(open('${result_file}'))
task = obj.get('results',{}).get('${TASK}',{})
for k in ['exact_match,flexible-extract','exact_match,strict-match','exact_match,none']:
    if k in task:
        print(f'{task[k]:.4f}')
        sys.exit(0)
print('NONE')
" 2>/dev/null || echo "NONE")
        if [ "$em" != "NONE" ]; then
            echo "  [CACHED] ${name}: EM = ${em}"
            DONE_NAMES+=("${name}")
            continue
        fi
    fi
    TODO_NAMES+=("${name}")
    TODO_PATHS+=("${EVAL_PATHS[$i]}")
done

echo ""
echo "Already done: ${#DONE_NAMES[@]}, To run: ${#TODO_NAMES[@]}"
echo ""

# Run evaluations in batches of 4 GPUs
NGPUS=4
TODO_COUNT=${#TODO_NAMES[@]}
NBATCH=$(( (TODO_COUNT + NGPUS - 1) / NGPUS ))

for batch in $(seq 0 $((NBATCH - 1))); do
    start=$((batch * NGPUS))
    echo "--- Eval batch $((batch+1))/${NBATCH} ---"

    PIDS=()
    for offset in $(seq 0 $((NGPUS - 1))); do
        idx=$((start + offset))
        if [ $idx -ge $TODO_COUNT ]; then break; fi

        name=${TODO_NAMES[$idx]}
        model_path=${TODO_PATHS[$idx]}
        gpu=$offset
        eval_dir="${EVAL_DIR}/${name}"

        rm -rf "${eval_dir}" 2>/dev/null
        eval_one "${model_path}" "${eval_dir}" ${gpu} "${name}" &
        PIDS+=($!)
        sleep 5
    done

    for pid in "${PIDS[@]}"; do
        wait $pid || true
    done
    echo ""
done

# ---------------------------------------------------------------
# PHASE 3: Aggregate all results into all_results.json
# ---------------------------------------------------------------
echo "=== PHASE 3: Aggregating results ==="
echo ""

$PYTHON -u - <<'PYEOF'
import json, os, sys
from pathlib import Path

SWEEP = Path("/mnt/beegfs/youyang7/projects/LostInSecond/artifacts/full_ft_sweep")
EVAL_DIR = SWEEP / "_gsm8k_vllm_eval"
TASK = "gsm8k_cot_zeroshot_unified"

def extract_em(results_json):
    obj = json.loads(results_json.read_text())
    task_block = (obj.get("results") or {}).get(TASK, {})
    for key in ["exact_match,flexible-extract", "exact_match,strict-match",
                "exact_match,none", "exact_match"]:
        if key in task_block:
            try:
                return float(task_block[key])
            except Exception:
                pass
    return None

def load_training_metrics(exp_dir):
    mf = exp_dir / "sweep_metrics.json"
    if not mf.exists():
        return {}
    m = json.loads(mf.read_text())
    cfg = m.get("config", {})
    log_history = m.get("log_history", [])
    eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]
    best_eval = min(eval_losses) if eval_losses else m.get("eval_metrics", {}).get("eval_loss")
    return {
        "eval_loss": m.get("eval_metrics", {}).get("eval_loss"),
        "best_eval_loss": best_eval,
        "train_loss": m.get("train_metrics", {}).get("train_loss"),
        "lr": cfg.get("learning_rate"),
        "epochs": cfg.get("num_train_epochs"),
        "wd": cfg.get("weight_decay"),
        "warmup": cfg.get("warmup_ratio"),
    }

results = {}

for d in sorted(EVAL_DIR.iterdir()):
    if not d.is_dir():
        continue
    name = d.name
    rfiles = sorted(d.rglob("results_*.json"))
    if not rfiles:
        continue
    em = extract_em(rfiles[-1])
    entry = {"name": name, "gsm8k_em": em, "method": "base" if name == "base" else "full_ft"}
    if name != "base":
        training = load_training_metrics(SWEEP / name)
        entry.update(training)
    results[name] = entry

out_file = EVAL_DIR / "all_results.json"
out_file.write_text(json.dumps(list(results.values()), indent=2, default=str))
print(f"Aggregated {len(results)} results -> {out_file}")

base_em = results.get("base", {}).get("gsm8k_em")
print(f"\nBase model: GSM8K EM = {base_em}")
print(f"\n{'Rk':<4} {'Name':<20} {'GSM8K EM':<10} {'Delta':<8} {'EvalLoss':<10} {'LR':<8} {'Ep':<5} {'WD':<6}")
print("-" * 80)

ft = sorted([(n, r) for n, r in results.items() if n != "base"],
            key=lambda x: -(x[1].get("gsm8k_em") or -1))

for i, (name, r) in enumerate(ft):
    em = r.get("gsm8k_em")
    em_str = f"{em:.4f}" if em is not None else "FAIL"
    delta = f"{em - base_em:+.4f}" if (em is not None and base_em is not None) else "N/A"
    el = f"{r.get('eval_loss', 0):.4f}" if r.get('eval_loss') is not None else "-"
    lr = f"{r.get('lr', 0):.0e}" if r.get('lr') is not None else "-"
    ep = str(r.get('epochs', '-'))
    wd = str(r.get('wd', '-'))
    print(f"{i+1:<4} {name:<20} {em_str:<10} {delta:<8} {el:<10} {lr:<8} {ep:<5} {wd:<6}")

ok = [(n, r) for n, r in ft if r.get("gsm8k_em") is not None]
if ok and base_em:
    improved = sum(1 for _, r in ok if r["gsm8k_em"] > base_em)
    print(f"\n{improved}/{len(ok)} experiments beat base model")
    best_name, best_r = ok[0]
    print(f"Best full FT: {best_name} (EM={best_r['gsm8k_em']:.4f}, delta={best_r['gsm8k_em'] - base_em:+.4f})")

# Also load LoRA best for comparison
lora_file = Path("/mnt/beegfs/youyang7/projects/LostInSecond/artifacts/lora_sweep/_gsm8k_vllm_eval/all_results.json")
if lora_file.exists():
    lora_all = json.loads(lora_file.read_text())
    lora_adapters = [r for r in lora_all if r.get("name") != "base" and r.get("gsm8k_em")]
    if lora_adapters:
        best_lora = max(lora_adapters, key=lambda x: x["gsm8k_em"])
        print(f"\nBest LoRA: {best_lora['name']} (EM={best_lora['gsm8k_em']:.4f})")
        if ok:
            delta = ok[0][1]['gsm8k_em'] - best_lora['gsm8k_em']
            print(f"Full FT vs LoRA: {delta:+.4f}")
PYEOF

# Also run the report generator from 14_full_ft_sweep_and_eval.py
echo ""
echo "Generating markdown report..."
$PYTHON -u "$EVAL_SCRIPT" --phase summary-only 2>&1 || echo "[WARN] Report generation returned non-zero"

echo ""
echo "============================================================"
echo "  ALL DONE: $(date)"
echo "============================================================"
