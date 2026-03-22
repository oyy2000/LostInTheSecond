#!/bin/bash
set -e
cd /jet/home/swang47/yang/projects/LostInTheSecond

LORA_JOB=$(sbatch scripts/slurm/31_single_train.sh lora_gpt4o_combined lora artifacts_real/full_gpt4o/lemma_sft_combined_gpt4o.json 2e-4 3 16 0.03 0.01 2>&1)
echo "LoRA: $LORA_JOB"

FT_JOB=$(sbatch scripts/slurm/31_single_train.sh ft_gpt4o_combined full_ft artifacts_real/full_gpt4o/lemma_sft_combined_gpt4o.json 2e-5 3 16 0.03 0.01 2>&1)
echo "Full FT: $FT_JOB"

echo "Both submitted."
