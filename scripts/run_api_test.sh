#!/bin/bash
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang
cd /jet/home/swang47/yang/projects/LostInTheSecond
python scripts/test_api.py > /tmp/api_test_result.txt 2>&1
echo "EXIT_CODE=$?" >> /tmp/api_test_result.txt
