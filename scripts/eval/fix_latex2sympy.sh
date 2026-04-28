#!/bin/bash
set -x
LOGFILE=/jet/home/swang47/yang/projects/LostInTheSecond/logs/fix_latex2sympy.log
exec > "$LOGFILE" 2>&1

PIP=/ocean/projects/cis250050p/swang47/miniconda3/envs/sft_yang/bin/pip
PY=/ocean/projects/cis250050p/swang47/miniconda3/envs/sft_yang/bin/python3

echo "=== Uninstall ==="
$PIP uninstall -y latex2sympy2

echo "=== Reinstall ==="
cd /jet/home/swang47/yang/projects/LEMMA/evaluation/latex2sympy
$PIP install -e .

echo "=== Verify ==="
$PY -c "from latex2sympy2 import latex2sympy; print('IMPORT OK')"

echo "=== Submit eval ==="
sbatch /jet/home/swang47/yang/projects/LostInTheSecond/scripts/slurm/submit_eval_models.slurm

echo "=== DONE ==="
