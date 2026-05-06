#!/bin/bash
# Batch re-run: recount tokens + re-eval for all datasets and models
# Run from project root: bash scripts/batch_reeval.sh

set -e

PROJ="/common/users/sl2148/Public/yang_ouyang/projects/LostInTheSecond"
PYTHON_TK="/common/users/sl2148/anaconda3/envs/fact_yang/bin/python"
PYTHON_EVAL="python"
SIGNALS="nll_drop_fb_last prm_drop_fb_last"

cd "$PROJ"

# ---- Model configs: result_prefix -> HuggingFace model ID ----
declare -A MODELS=(
  # ["qwen2.5_3b_instruct_budget_multisignal"]="Qwen/Qwen2.5-3B-Instruct"
  ["llama_3.2_3b_instruct_budget_multisignal"]="meta-llama/Llama-3.2-3B-Instruct"
  # ["deepseek_r1_distill_qwen_7b_budget_multisignal"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

)

# ---- Dataset dirs -> --dataset argument ----
declare -A DS_MAP=(
  # ["math500"]="math500"
  # ["gsm8k"]="gsm8k"
  # ["amc2023"]="amc2023"
  # ["aime2024"]="aime2024"
  # ["aime2025"]="aime2025"
  # ["olympiadbench"]="olympiadbench"
  ["hotpotqa_open"]="hotpotqa_open"
  ["2wikimultihopqa_open"]="2wikimultihopqa_open"
)

for model_dir in "${!MODELS[@]}"; do
  model_id="${MODELS[$model_dir]}"
  RESULTS="$PROJ/results/$model_dir"

  if [ ! -d "$RESULTS" ]; then
    echo "SKIP model dir: $model_dir (not found)"
    continue
  fi

  echo "========================================"
  echo "MODEL: $model_dir"
  echo "  HF ID: $model_id"
  echo "========================================"

  for ds_dir in "${!DS_MAP[@]}"; do
    dataset="${DS_MAP[$ds_dir]}"
    ckpt="$RESULTS/$ds_dir/checkpoint.jsonl"

    if [ ! -f "$ckpt" ]; then
      continue
    fi

    echo ""
    echo "--- $model_dir / $ds_dir (dataset=$dataset) ---"

    # Infer n_sample from checkpoint (count unique doc_ids in draft records)
    N_SAMPLE=$(grep '"task_type": "draft"' "$ckpt" \
      | python3 -c "
import sys, json
ids = set()
for line in sys.stdin:
    r = json.loads(line)
    ids.add(r['doc_id'])
print(len(ids))
")
    echo "  Detected $N_SAMPLE questions from checkpoint"

    # Backup eval_summary_table.md
    if [ -f "$RESULTS/$ds_dir/eval_summary_table.md" ]; then
      cp "$RESULTS/$ds_dir/eval_summary_table.md" \
         "$RESULTS/$ds_dir/eval_summary_table_old.md"
      echo "  Backed up eval_summary_table.md"
    fi

    # Backup eval_summary.json
    if [ -f "$RESULTS/$ds_dir/eval_summary.json" ]; then
      cp "$RESULTS/$ds_dir/eval_summary.json" \
         "$RESULTS/$ds_dir/eval_summary_old_2.json"
      echo "  Backed up eval_summary.json"
    fi

    # Step 1: Recount tokens
    echo "  Step 1: Recount tokens..."
    $PYTHON_TK scripts/6_8_1_recount_tokens.py \
      --checkpoint "$ckpt" \
      --model "$model_id"

    # Step 2: Re-run eval
    echo "  Step 2: Re-run eval..."
    $PYTHON_EVAL scripts/6_8_budget_controlled_multisignal.py \
      --dataset "$dataset" --budget 32 --gpus "" \
      --signals $SIGNALS \
      --n-sample "$N_SAMPLE" \
      --skip-phase 1 2 25 3 4 5 \
      --out-dir "$RESULTS/$ds_dir"

    echo "  Done: $ds_dir"
  done
done

echo ""
echo "All models and datasets processed."