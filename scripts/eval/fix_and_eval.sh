#!/bin/bash
set -x

PIP=/ocean/projects/cis250050p/swang47/miniconda3/envs/sft_yang/bin/pip
PY=/ocean/projects/cis250050p/swang47/miniconda3/envs/sft_yang/bin/python3

echo "=== Step 1: Uninstall broken latex2sympy2 ==="
$PIP uninstall -y latex2sympy2

echo "=== Step 2: Reinstall from correct path ==="
cd /jet/home/swang47/yang/projects/LEMMA/evaluation/latex2sympy
$PIP install -e .

echo "=== Step 3: Verify ==="
$PY -c "from latex2sympy2 import latex2sympy; print('latex2sympy2 import OK')"

echo "=== Step 4: Submit eval job ==="
sbatch /jet/home/swang47/yang/projects/LostInTheSecond/scripts/slurm/submit_eval_models.slurm

echo "=== DONE ==="
