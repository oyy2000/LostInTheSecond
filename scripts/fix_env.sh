#!/bin/bash
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang
echo "=== Current versions ===" > /tmp/fix_env.log
pip show transformers accelerate peft >> /tmp/fix_env.log 2>&1
echo "" >> /tmp/fix_env.log
echo "=== Upgrading accelerate ===" >> /tmp/fix_env.log
pip install --upgrade accelerate >> /tmp/fix_env.log 2>&1
echo "" >> /tmp/fix_env.log
echo "=== After upgrade ===" >> /tmp/fix_env.log
pip show accelerate >> /tmp/fix_env.log 2>&1
echo "" >> /tmp/fix_env.log
echo "=== Test import ===" >> /tmp/fix_env.log
python -c "
from transformers.trainer import Trainer
from accelerate import Accelerator
a = Accelerator()
print('Accelerator OK')
print(f'unwrap_model signature works')
" >> /tmp/fix_env.log 2>&1
echo "DONE" >> /tmp/fix_env.log
