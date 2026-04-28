#!/bin/bash
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang
pip uninstall autoawq autoawq-kernels -y 2>&1
echo "Done. awq packages removed."
python -c "from transformers.trainer import Trainer; print('Trainer import OK')" 2>&1
