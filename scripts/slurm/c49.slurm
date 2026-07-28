#!/bin/bash

#SBATCH -J 6_8_bms_llama
#SBATCH -p a5000ada
#SBATCH -N 1
#SBATCH -w c49
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/projects/LostInTheSecond/logs/%x/%x-%j.out
#SBATCH --error=/home/%u/projects/LostInTheSecond/logs/%x/%x-%j.err

set -euo pipefail

source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
conda activate fact

cd /home/youyang7/projects/LostInTheSecond

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="/home/${USER}/projects/.cache/hf_datasets/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

mkdir -p "/home/${USER}/projects/LostInTheSecond/logs/${SLURM_JOB_NAME}"
mkdir -p "${HF_DATASETS_CACHE}"

echo "Running on $(hostname)"
which python
python -V
nvidia-smi

python -u scripts/6_9_sweep_gpt_signals.py \
    --gpus 1,2,3 \
    --models llama3b \
    --datasets olympiadbench \
    --signals prm_drop_fb_last nll_drop_fb_last \
    --n_samples 500 \
    --skip-phase 25 \
    --nd 4 8 16 
