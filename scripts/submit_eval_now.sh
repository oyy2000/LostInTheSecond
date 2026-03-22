#!/bin/bash
exec > /tmp/submit_eval_now.log 2>&1
set -x
date
cd /jet/home/swang47/yang/projects/LostInTheSecond
sbatch scripts/slurm/30_lemma_eval_gsm8k.sh eval-lora-gpt4o /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/artifacts/full_prefill_llama8b/lora_gpt4o_combined/merged_model
squeue -u swang47
date
echo "SCRIPT_DONE"
