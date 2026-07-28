#!/bin/bash
#SBATCH -J rebuttal_qwen_math
#SBATCH -p a5000ada
#SBATCH -N 1
#SBATCH -w c49
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/%x-%j.out
#SBATCH --error=/home/youyang7/projects/LostInTheSecond/rebuttal/logs/%x-%j.err

set -euo pipefail

if [ -f /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh ]; then
    source /mnt/beegfs/youyang7/miniconda3/etc/profile.d/conda.sh
else
    source /home/youyang7/miniconda3/etc/profile.d/conda.sh
fi
conda activate fact

cd /home/youyang7/projects/LostInTheSecond

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="/tmp/${USER}_${SLURM_JOB_ID}_hf_datasets"
export MPLCONFIGDIR="/tmp/${USER}_${SLURM_JOB_ID}_matplotlib"
mkdir -p "${HF_DATASETS_CACHE}" "${MPLCONFIGDIR}" rebuttal/logs

wait_for_gpus() {
    local threshold_mib=12500
    while true; do
        local free_count
        free_count=$(nvidia-smi \
            --query-gpu=memory.free \
            --format=csv,noheader,nounits \
            | awk -v threshold="${threshold_mib}" \
                '$1 >= threshold {count++} END {print count+0}')
        if [ "${free_count}" -ge 3 ]; then
            break
        fi
        echo "Waiting for 3 GPUs above ${threshold_mib} MiB: ${free_count}/3"
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader
        sleep 300
    done
}

echo "Host: $(hostname)"
wait_for_gpus
nvidia-smi

python -u rebuttal/scripts/6_9_qwen_position_controls.py \
    --dataset math500 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --gpus 0,1,2 \
    --n-sample 500 \
    --dataset-seed 42 \
    --random-seed 42 \
    --generation-seed 42 \
    --gpu-memory-utilization 0.32 \
    --budget 32 \
    --nd 4 8 16 \
    --strategies \
        random_uniform fixed_25 fixed_50 fixed_75 final_step_only \
    --reuse-checkpoint \
        rebuttal/results_old/qwen2.5_3b_instruct_position_controls/seed_42/math500/checkpoint.jsonl \
    --out-dir \
        rebuttal/results/qwen2.5_3b_instruct_position_controls_final_step/seed_42/math500
