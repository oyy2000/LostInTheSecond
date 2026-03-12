#!/bin/bash

PYTHON=/mnt/beegfs/youyang7/.conda/envs/fact/bin/python
SCRIPT=/home/youyang7/projects/LostInTheSecond/scripts/full_ft/13_full_finetune.py
DATASET=/mnt/beegfs/youyang7/projects/LostInSecond/artifacts/samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json
SWEEP=/mnt/beegfs/youyang7/projects/LostInSecond/artifacts/full_ft_sweep
MODEL=Qwen/Qwen2.5-3B-Instruct

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_exp() {
    local name=$1 lr=$2 epochs=$3 wd=$4 warmup=$5 gpu=$6
    local outdir="${SWEEP}/${name}"

    if [ -f "${outdir}/best_model/config.json" ] && [ -f "${outdir}/sweep_metrics.json" ]; then
        echo "[SKIP] ${name} already complete"
        return 0
    fi

    rm -rf "${outdir}" 2>/dev/null
    echo "[RUN] ${name} (lr=${lr}, ep=${epochs}, wd=${wd}, warmup=${warmup}) on GPU ${gpu}"

    CUDA_VISIBLE_DEVICES=${gpu} $PYTHON -u $SCRIPT \
        --model-id $MODEL \
        --dataset-path $DATASET \
        --output-dir "${outdir}" \
        --learning-rate ${lr} \
        --num-train-epochs ${epochs} \
        --weight-decay ${wd} \
        --warmup-ratio ${warmup} \
        --gradient-accumulation-steps 16 \
        --per-device-train-batch-size 1 \
        --logging-steps 1 --eval-steps 7 --save-steps 7 --save-total-limit 2 --seed 42

    if [ -f "${outdir}/best_model/config.json" ]; then
        echo "[OK] ${name} completed"
    else
        echo "[FAIL] ${name}"
    fi
}

echo "=== Running remaining full FT experiments sequentially ==="
echo "Started: $(date)"

# Group A: LR sweep (missing: 1e-6, 1e-5, 2e-5, 5e-5)
run_exp ft_lr1e6  1e-6  3 0.01 0.1 0
run_exp ft_lr1e5  1e-5  3 0.01 0.1 0
run_exp ft_lr2e5  2e-5  3 0.01 0.1 0
run_exp ft_lr5e5  5e-5  3 0.01 0.1 0

# Group B: Epoch sweep (missing: ep5)
run_exp ft_ep5    1e-5  5 0.01 0.1 0

# Group D: Combined (missing: combo_a)
run_exp ft_combo_a 2e-5 2 0.01 0.1 0

echo ""
echo "=== All experiments done ==="
echo "Finished: $(date)"

echo ""
echo "=== Summary of all experiments ==="
for d in ${SWEEP}/ft_*/; do
    name=$(basename "$d")
    if [ -f "$d/best_model/config.json" ] && [ -f "$d/sweep_metrics.json" ]; then
        echo "  COMPLETE: $name"
    else
        echo "  INCOMPLETE: $name"
    fi
done
