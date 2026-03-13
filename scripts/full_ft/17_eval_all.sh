#!/bin/bash
set -euo pipefail

PYTHON=/mnt/beegfs/youyang7/.conda/envs/fact/bin/python
EVAL_SCRIPT=/home/youyang7/projects/LostInTheSecond/scripts/full_ft/14_full_ft_sweep_and_eval.py
SWEEP=/mnt/beegfs/youyang7/projects/LostInSecond/artifacts/full_ft_sweep
EVAL_DIR=${SWEEP}/_gsm8k_vllm_eval
MODEL=Qwen/Qwen2.5-3B-Instruct
HARNESS_DIR=/home/youyang7/projects/LostInTheSecond/lm-evaluation-harness
TASK=gsm8k_cot_zeroshot_unified

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

VLLM_ARGS="pretrained=__MODEL__,dtype=bfloat16,gpu_memory_utilization=0.55,max_model_len=3072,max_num_seqs=16,enforce_eager=True"

echo "============================================================"
echo "  GSM8K EVALUATION FOR ALL FULL FT MODELS"
echo "  Started: $(date)"
echo "============================================================"

eval_one() {
    local model_path=$1 output_dir=$2 gpu=$3 name=$4

    mkdir -p "${output_dir}"
    local model_args
    model_args=$(echo "$VLLM_ARGS" | sed "s|__MODEL__|${model_path}|g")

    echo "[EVAL] ${name} on GPU ${gpu} -- $(date +%H:%M:%S)"
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
        echo "[OK] ${name}: GSM8K EM = ${em} -- $(date +%H:%M:%S)"
    else
        echo "[FAIL] ${name}: no results file -- check ${output_dir}/eval.log"
    fi
}

# ---------------------------------------------------------------
# Build model list and check for cached results
# ---------------------------------------------------------------
declare -a ALL_NAMES=()
declare -a ALL_PATHS=()

ALL_NAMES+=("base")
ALL_PATHS+=("${MODEL}")

for d in ${SWEEP}/ft_*/; do
    name=$(basename "$d")
    model_path="${d}best_model"
    if [ -f "${model_path}/config.json" ]; then
        ALL_NAMES+=("${name}")
        ALL_PATHS+=("${model_path}")
    fi
done

echo "Total models found: ${#ALL_NAMES[@]}"
echo ""

declare -a TODO_NAMES=()
declare -a TODO_PATHS=()

for i in $(seq 0 $((${#ALL_NAMES[@]} - 1))); do
    name=${ALL_NAMES[$i]}
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
            continue
        fi
    fi
    TODO_NAMES+=("${name}")
    TODO_PATHS+=("${ALL_PATHS[$i]}")
done

echo ""
echo "To evaluate: ${#TODO_NAMES[@]} models"
echo ""

# ---------------------------------------------------------------
# Run evaluations: 4 GPUs, one model per GPU per batch
# Stagger launches by 15s to avoid simultaneous vLLM startup
# ---------------------------------------------------------------
NGPUS=4
TODO_COUNT=${#TODO_NAMES[@]}

if [ $TODO_COUNT -eq 0 ]; then
    echo "All evaluations already cached!"
else
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
            sleep 15
        done

        for pid in "${PIDS[@]}"; do
            wait $pid || echo "[WARN] Eval PID $pid failed"
        done
        echo ""
    done
fi

# ---------------------------------------------------------------
# Aggregate all results
# ---------------------------------------------------------------
echo "=== Aggregating results ==="
echo ""

$PYTHON -u - <<'PYEOF'
import json, time
from pathlib import Path

SWEEP = Path("/mnt/beegfs/youyang7/projects/LostInSecond/artifacts/full_ft_sweep")
EVAL_DIR = SWEEP / "_gsm8k_vllm_eval"
LORA_EVAL = Path("/mnt/beegfs/youyang7/projects/LostInSecond/artifacts/lora_sweep/_gsm8k_vllm_eval/all_results.json")
DOCS_DIR = Path("/home/youyang7/projects/LostInTheSecond/documents")
TASK = "gsm8k_cot_zeroshot_unified"

def extract_em(rfile):
    obj = json.loads(rfile.read_text())
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
        "train_samples": cfg.get("train_samples"),
    }

# Collect results
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
        entry.update(load_training_metrics(SWEEP / name))
    results[name] = entry

out_file = EVAL_DIR / "all_results.json"
out_file.write_text(json.dumps(list(results.values()), indent=2, default=str))

base_em = results.get("base", {}).get("gsm8k_em")

# Print results table
print(f"\n{'='*90}")
print(f"{'FULL FT — GSM8K EVALUATION RESULTS':^90}")
print(f"{'='*90}")
print(f"\nBase model: Qwen/Qwen2.5-3B-Instruct  |  GSM8K EM = {base_em}")
print(f"\n{'Rk':<4} {'Name':<20} {'GSM8K EM':<10} {'Delta':<8} {'EvalLoss':<10} {'BestEL':<10} {'LR':<8} {'Ep':<5} {'WD':<6}")
print("-" * 90)

ft = sorted([(n, r) for n, r in results.items() if n != "base"],
            key=lambda x: -(x[1].get("gsm8k_em") or -1))

for i, (name, r) in enumerate(ft):
    em = r.get("gsm8k_em")
    em_str = f"{em:.4f}" if em is not None else "FAIL"
    delta = f"{em - base_em:+.4f}" if (em is not None and base_em is not None) else "N/A"
    el = f"{r.get('eval_loss', 0):.4f}" if r.get('eval_loss') is not None else "-"
    bel = f"{r.get('best_eval_loss', 0):.4f}" if r.get('best_eval_loss') is not None else "-"
    lr = f"{r.get('lr', 0):.0e}" if r.get('lr') is not None else "-"
    ep = str(r.get('epochs', '-'))
    wd = str(r.get('wd', '-'))
    print(f"{i+1:<4} {name:<20} {em_str:<10} {delta:<8} {el:<10} {bel:<10} {lr:<8} {ep:<5} {wd:<6}")

ok = [(n, r) for n, r in ft if r.get("gsm8k_em") is not None]
if ok and base_em:
    improved = sum(1 for _, r in ok if r["gsm8k_em"] > base_em)
    print(f"\n{improved}/{len(ok)} experiments beat base model")
    print(f"Best full FT: {ok[0][0]} (EM={ok[0][1]['gsm8k_em']:.4f}, delta={ok[0][1]['gsm8k_em'] - base_em:+.4f})")

# LoRA comparison
if LORA_EVAL.exists():
    lora_all = json.loads(LORA_EVAL.read_text())
    lora_adapters = [r for r in lora_all if r.get("name") != "base" and r.get("gsm8k_em")]
    if lora_adapters:
        best_lora = max(lora_adapters, key=lambda x: x["gsm8k_em"])
        print(f"\nBest LoRA: {best_lora['name']} (EM={best_lora['gsm8k_em']:.4f})")
        if ok:
            delta_vs_lora = ok[0][1]['gsm8k_em'] - best_lora['gsm8k_em']
            verdict = "BEATS" if delta_vs_lora > 0 else "LOSES TO"
            print(f"Full FT {verdict} LoRA by {delta_vs_lora:+.4f}")

# Generate markdown report
lines = [
    "# Full Fine-Tuning Sweep — GSM8K Evaluation Results",
    "",
    f"**Base Model**: `Qwen/Qwen2.5-3B-Instruct`",
    f"**Method**: Full parameter fine-tuning (no LoRA)",
    f"**Optimizer**: paged_adamw_8bit (bitsandbytes)",
    f"**Task**: `{TASK}` (GSM8K zero-shot CoT, 1319 test samples)",
    f"**Training Data**: ~109 train / ~6 eval samples",
    f"**Evaluation Backend**: vLLM (bf16, greedy decoding, max_gen_toks=2048)",
    f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
    "",
]

if base_em is not None:
    lines.append(f"## Base Model: GSM8K EM = **{base_em:.4f}**")
lines.append("")

if LORA_EVAL.exists():
    lora_all = json.loads(LORA_EVAL.read_text())
    lora_adapters = [r for r in lora_all if r.get("name") != "base" and r.get("gsm8k_em")]
    if lora_adapters:
        best_lora = max(lora_adapters, key=lambda x: x["gsm8k_em"])
        lines.append(f"## Best LoRA (comparison): `{best_lora['name']}` EM = **{best_lora['gsm8k_em']:.4f}**")
        lines.append("")

lines.extend([
    "## Full Results (Ranked by GSM8K Exact Match)",
    "",
    "| Rank | Name | GSM8K EM | Delta | Eval Loss | Best EL | Train Loss | LR | Epochs | WD | Warmup |",
    "|------|------|----------|-------|-----------|---------|------------|----|--------|----|--------|",
])

for i, (name, r) in enumerate(ft):
    em = r.get("gsm8k_em")
    em_str = f"{em:.4f}" if em is not None else "FAIL"
    delta = (em - base_em) if (em is not None and base_em is not None) else None
    delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
    el = f"{r.get('eval_loss'):.4f}" if r.get('eval_loss') is not None else "—"
    bel = f"{r.get('best_eval_loss'):.4f}" if r.get('best_eval_loss') is not None else "—"
    tl = f"{r.get('train_loss'):.4f}" if r.get('train_loss') is not None else "—"
    lr_val = r.get("lr")
    lr_str = f"{lr_val:.0e}" if lr_val is not None else "—"
    lines.append(
        f"| {i+1} | {name} | {em_str} | {delta_str} | {el} | {bel} | {tl} "
        f"| {lr_str} | {r.get('epochs', '—')} | {r.get('wd', '—')} | {r.get('warmup', '—')} |"
    )

if ok and base_em is not None:
    best_name, best_r = ok[0]
    worst_name, worst_r = ok[-1]
    improved = [x for x in ok if x[1]["gsm8k_em"] > base_em]
    lines.extend(["", "## Key Findings", ""])
    lines.append(f"- **Best full FT**: `{best_name}` (EM = {best_r['gsm8k_em']:.4f}, delta = {best_r['gsm8k_em'] - base_em:+.4f})")
    lines.append(f"- **Worst full FT**: `{worst_name}` (EM = {worst_r['gsm8k_em']:.4f}, delta = {worst_r['gsm8k_em'] - base_em:+.4f})")
    lines.append(f"- **{len(improved)}/{len(ok)}** experiments improved over base")

lines.extend(["", "---", f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*", ""])

report_path = DOCS_DIR / "full_ft_gsm8k_results.md"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"\nReport: {report_path}")
print(f"JSON:   {out_file}")
PYEOF

echo ""
echo "============================================================"
echo "  ALL DONE: $(date)"
echo "============================================================"
