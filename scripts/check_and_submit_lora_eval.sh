#!/bin/bash
LOG=/tmp/check_and_submit_lora_eval.log
exec > $LOG 2>&1
set -x

echo "=== SQUEUE ==="
squeue -u swang47 2>&1

echo "=== Check LoRA merged model ==="
ls -la /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/artifacts/full_prefill_llama8b/lora_gpt4o_combined/merged_model/config.json 2>&1

echo "=== Submit LoRA eval ==="
cd /jet/home/swang47/yang/projects/LostInTheSecond
sbatch scripts/slurm/30_lemma_eval_gsm8k.sh eval-lora-gpt4o /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/artifacts/full_prefill_llama8b/lora_gpt4o_combined/merged_model 2>&1

echo "=== DONE ==="
