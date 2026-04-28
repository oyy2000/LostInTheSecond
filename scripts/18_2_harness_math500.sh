#!/usr/bin/env bash
# Phase 18: Run hendrycks_math_500 via lm-evaluation-harness with HF and vLLM backends.
#
# Usage:
#   bash scripts/18_2_harness_math500.sh hf   [gpu_ids]   # e.g. bash ... hf 4
#   bash scripts/18_2_harness_math500.sh vllm [gpu_ids]   # e.g. bash ... vllm 4,5,6,7
#   bash scripts/18_2_harness_math500.sh both [gpu_ids]
#
# Outputs go to results/harness_math500_{hf,vllm}/

set -euo pipefail

BACKEND="${1:-vllm}"
GPUS="${2:-4,5,6,7}"
MODEL="meta-llama/Llama-3.2-3B-Instruct"
MAX_GEN_TOKS=4096
HARNESS_DIR="lm-evaluation-harness"
TASK_DIR="$(pwd)/${HARNESS_DIR}/lm_eval/tasks/hendrycks_math_500"
OUT_BASE="results/harness_math500"

# Number of GPUs for vLLM data_parallel_size
N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

export CUDA_VISIBLE_DEVICES="$GPUS"
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"

run_hf() {
    local out_dir="${OUT_BASE}_hf"
    local hf_gpu
    hf_gpu=$(echo "$GPUS" | cut -d',' -f1)
    echo "========================================"
    echo "  Backend : HF (transformers)"
    echo "  Model   : $MODEL"
    echo "  GPU     : $hf_gpu"
    echo "  max_gen : $MAX_GEN_TOKS"
    echo "  Output  : $out_dir"
    echo "========================================"

    # Use --device cuda:N (not device_map in model_args) and no --apply_chat_template,
    # matching the working configuration. The task yaml already includes the full prompt.
    PYTHONPATH="${HARNESS_DIR}:${PYTHONPATH:-}" python -m lm_eval \
        --model hf \
        --model_args "pretrained=${MODEL},dtype=float16" \
        --tasks hendrycks_math_500 \
        --include_path "${TASK_DIR}" \
        --device "cuda:${hf_gpu}" \
        --gen_kwargs "do_sample=False,temperature=0,max_gen_toks=${MAX_GEN_TOKS}" \
        --batch_size auto \
        --log_samples \
        --output_path "$out_dir" \
        --trust_remote_code \
        2>&1 | tee "${out_dir}_run.log"

    echo "HF done. Results in $out_dir"
}

run_vllm() {
    local out_dir="${OUT_BASE}_vllm"
    # Single GPU is fastest for 3B: no data-parallel overhead, vLLM handles
    # continuous batching internally.
    local vllm_gpu
    vllm_gpu=$(echo "$GPUS" | cut -d',' -f1)
    echo "========================================"
    echo "  Backend : vLLM"
    echo "  Model   : $MODEL"
    echo "  GPU     : $vllm_gpu  (single card)"
    echo "  max_gen : $MAX_GEN_TOKS"
    echo "  Output  : $out_dir"
    echo "========================================"

    CUDA_VISIBLE_DEVICES="$vllm_gpu" PYTHONPATH="${HARNESS_DIR}:${PYTHONPATH:-}" python -m lm_eval \
        --model vllm \
        --model_args "pretrained=${MODEL},dtype=half,gpu_memory_utilization=0.85,tensor_parallel_size=1,max_model_len=$((MAX_GEN_TOKS + 512)),max_num_seqs=256,trust_remote_code=True" \
        --tasks hendrycks_math_500 \
        --include_path "${TASK_DIR}" \
        --gen_kwargs "do_sample=false,temperature=0,max_gen_toks=${MAX_GEN_TOKS}" \
        --apply_chat_template \
        --batch_size auto \
        --log_samples \
        --output_path "$out_dir" \
        --trust_remote_code \
        2>&1 | tee "${out_dir}_run.log"

    echo "vLLM done. Results in $out_dir"
}

mkdir -p "$OUT_BASE"_hf "$OUT_BASE"_vllm

case "$BACKEND" in
    hf)   run_hf ;;
    vllm) run_vllm ;;
    both) run_hf; run_vllm ;;
    *)    echo "Unknown backend: $BACKEND. Use hf, vllm, or both."; exit 1 ;;
esac

# Print accuracy summary
echo ""
echo "========================================"
echo "  ACCURACY SUMMARY"
echo "========================================"
python3 - <<'PYEOF'
import json, glob, sys
from pathlib import Path

for backend in ("hf", "vllm"):
    pattern = f"results/harness_math500_{backend}/**/*.json"
    files = sorted(glob.glob(pattern, recursive=True))
    result_files = [f for f in files if "results_" in Path(f).name]
    if not result_files:
        print(f"  {backend.upper()}: no results file found yet")
        continue
    latest = result_files[-1]
    d = json.load(open(latest))
    tasks = d.get("results", {})
    for task, metrics in tasks.items():
        acc = metrics.get("exact_match,none", metrics.get("exact_match", "?"))
        print(f"  {backend.upper()} | {task}: exact_match = {acc}")
PYEOF
