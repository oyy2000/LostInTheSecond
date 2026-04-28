#!/bin/bash
LOG=/tmp/fix_and_resubmit.log
exec > $LOG 2>&1
set -x

source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

echo "=== Current versions ==="
pip show accelerate transformers 2>&1 | grep -E "^(Name|Version)"

echo "=== Upgrading accelerate ==="
pip install --upgrade accelerate

echo "=== New version ==="
pip show accelerate 2>&1 | grep -E "^(Name|Version)"

echo "=== Test import ==="
python -c "from transformers.trainer import Trainer; print('Trainer import OK')"

echo "=== Resubmitting LoRA ==="
cd /jet/home/swang47/yang/projects/LostInTheSecond
sbatch scripts/slurm/31_single_train.sh lora_gpt4o_combined lora artifacts_real/full_gpt4o/lemma_sft_combined_gpt4o.json 2e-4 3 16 0.03 0.01

echo "=== Resubmitting Full FT ==="
sbatch scripts/slurm/31_single_train.sh ft_gpt4o_combined full artifacts_real/full_gpt4o/lemma_sft_combined_gpt4o.json 2e-5 3 16 0.03 0.01

echo "=== ALL DONE ==="
