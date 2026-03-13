#!/bin/bash
set -euo pipefail

echo "=== Step 1: Killing zombie training processes ==="

# Kill known zombie PIDs from previous runs
for pid in 2788735 2788736 2788800; do
    kill -9 $pid 2>/dev/null && echo "Killed PID $pid" || echo "PID $pid already dead"
done

# Kill any remaining python processes running 13_full_finetune.py
# But NOT the gg process (PID 2535010)
echo ""
echo "Scanning for any remaining training processes..."
ps aux | grep "13_full_finetune" | grep -v grep | while read -r line; do
    pid=$(echo "$line" | awk '{print $2}')
    echo "  Killing training process PID=$pid"
    kill -9 "$pid" 2>/dev/null || true
done

echo ""
echo "Waiting 10 seconds for GPU memory to be freed..."
sleep 10

echo ""
echo "=== Step 2: Checking GPU status ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo ""

# Verify GPUs are mostly free (just gg process ~1.3GB per GPU)
echo "GPU processes:"
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader 2>/dev/null || nvidia-smi | grep -A 10 "Processes:"

echo ""
echo "=== Step 3: Re-running 3 training experiments in parallel ==="

PYTHON=/mnt/beegfs/youyang7/.conda/envs/fact/bin/python
SCRIPT=/home/youyang7/projects/LostInTheSecond/scripts/full_ft/13_full_finetune.py
DATASET=/mnt/beegfs/youyang7/projects/LostInSecond/artifacts/samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json
SWEEP=/mnt/beegfs/youyang7/projects/LostInSecond/artifacts/full_ft_sweep
MODEL=Qwen/Qwen2.5-3B-Instruct

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

train_one() {
    local name=$1 lr=$2 gpu=$3
    local outdir="${SWEEP}/${name}"

    if [ -f "${outdir}/best_model/config.json" ] && [ -f "${outdir}/sweep_metrics.json" ]; then
        echo "[SKIP] ${name} already complete"
        return 0
    fi

    rm -rf "${outdir}" 2>/dev/null
    mkdir -p "${outdir}"
    echo "[TRAIN] ${name} (lr=${lr}) on GPU ${gpu} -- started $(date +%H:%M:%S)"

    CUDA_VISIBLE_DEVICES=${gpu} $PYTHON -u $SCRIPT \
        --model-id $MODEL \
        --dataset-path $DATASET \
        --output-dir "${outdir}" \
        --learning-rate ${lr} \
        --num-train-epochs 3 \
        --weight-decay 0.01 \
        --warmup-ratio 0.1 \
        --gradient-accumulation-steps 16 \
        --per-device-train-batch-size 1 \
        --logging-steps 1 --eval-steps 7 --save-steps 7 --save-total-limit 2 --seed 42 \
        > "${outdir}/training.log" 2>&1

    if [ -f "${outdir}/best_model/config.json" ]; then
        echo "[OK] ${name} complete -- $(date +%H:%M:%S)"
    else
        echo "[FAIL] ${name} -- check ${outdir}/training.log"
    fi
}

# Launch one per GPU, staggered by 30s to avoid simultaneous model loading
train_one ft_lr1e5 1e-5 0 &
PID1=$!
sleep 30

train_one ft_lr2e5 2e-5 1 &
PID2=$!
sleep 30

train_one ft_lr5e5 5e-5 2 &
PID3=$!

echo ""
echo "Training PIDs: $PID1 $PID2 $PID3"
echo "Waiting for all training jobs to complete..."

FAIL=0
for pid in $PID1 $PID2 $PID3; do
    wait $pid || FAIL=$((FAIL + 1))
done

echo ""
echo "=== Step 4: Training Results ==="
for d in ${SWEEP}/ft_*/; do
    name=$(basename "$d")
    if [ -f "${d}best_model/config.json" ] && [ -f "${d}sweep_metrics.json" ]; then
        echo "  COMPLETE: $name"
    else
        echo "  INCOMPLETE: $name"
    fi
done

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "[WARN] $FAIL training job(s) failed. Check logs:"
    for name in ft_lr1e5 ft_lr2e5 ft_lr5e5; do
        log="${SWEEP}/${name}/training.log"
        if [ -f "$log" ]; then
            has_error=$(grep -c "Error\|FAIL\|OOM" "$log" 2>/dev/null || echo 0)
            if [ "$has_error" -gt 0 ]; then
                echo "  $name: $(tail -3 "$log")"
            fi
        fi
    done
fi

echo ""
echo "=== Step 5: Ready for GSM8K evaluation ==="
echo "To run evaluation, execute:"
echo "  nohup bash /home/youyang7/projects/LostInTheSecond/scripts/full_ft/17_eval_all.sh > /mnt/beegfs/youyang7/projects/LostInSecond/logs/full_ft_eval.log 2>&1 &"
echo ""
echo "DONE: $(date)"
