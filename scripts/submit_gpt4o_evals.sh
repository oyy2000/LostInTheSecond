#!/bin/bash
set -e
cd /jet/home/swang47/yang/projects/LostInTheSecond

BASE=/ocean/projects/cis250050p/swang47/yang/LostInTheSecond/artifacts/full_prefill_llama8b

LORA_MERGED="$BASE/lora_gpt4o_combined/merged_model"
FT_BEST="$BASE/ft_gpt4o_combined/best_model"

LOG=/tmp/submit_evals.log

echo "=== Checking models ===" > $LOG

if [ -f "$LORA_MERGED/config.json" ]; then
    echo "LoRA merged model found" >> $LOG
    sbatch scripts/slurm/30_lemma_eval_gsm8k.sh eval-lora-gpt4o "$LORA_MERGED" >> $LOG 2>&1
else
    echo "LoRA merged model NOT found at $LORA_MERGED" >> $LOG
fi

if [ -f "$FT_BEST/config.json" ]; then
    echo "Full FT best model found" >> $LOG
    sbatch scripts/slurm/30_lemma_eval_gsm8k.sh eval-ft-gpt4o "$FT_BEST" >> $LOG 2>&1
else
    echo "Full FT best model NOT found at $FT_BEST" >> $LOG
fi

echo "=== DONE ===" >> $LOG
