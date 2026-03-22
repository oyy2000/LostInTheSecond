#!/bin/bash
set -e
cd /jet/home/swang47/yang/projects/LostInTheSecond

sbatch scripts/slurm/31_single_train.sh lora_gpt4o_combined lora artifacts_real/full_gpt4o/lemma_sft_combined_gpt4o.json 2e-4 3 16 0.03 0.01 > /tmp/resubmit_result.txt 2>&1
sbatch scripts/slurm/31_single_train.sh ft_gpt4o_combined full_ft artifacts_real/full_gpt4o/lemma_sft_combined_gpt4o.json 2e-5 3 16 0.03 0.01 >> /tmp/resubmit_result.txt 2>&1
echo "Done" >> /tmp/resubmit_result.txt
