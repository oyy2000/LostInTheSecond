#!/bin/bash
cd /jet/home/swang47/yang/projects/LostInTheSecond
MODEL_PATH="/ocean/projects/cis250050p/swang47/yang/LostInTheSecond/artifacts/full_prefill_llama8b/lora_gpt4o_combined/merged_model"

if [ -f "${MODEL_PATH}/config.json" ]; then
    echo "Model found at ${MODEL_PATH}"
    sbatch scripts/slurm/30_lemma_eval_gsm8k.sh eval-lora-gpt4o "${MODEL_PATH}" 2>&1
    echo "Submitted LoRA GPT-4o eval"
else
    echo "ERROR: Model not found at ${MODEL_PATH}"
fi
