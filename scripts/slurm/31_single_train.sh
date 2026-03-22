#!/bin/bash
#SBATCH --job-name=fp-train
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --account=cis250050p
#SBATCH --output=/jet/home/swang47/yang/projects/LostInTheSecond/logs/fp_train_%j.out
#SBATCH --error=/jet/home/swang47/yang/projects/LostInTheSecond/logs/fp_train_%j.err

set -euo pipefail

module load cuda/12.6.1
source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

export HF_TOKEN=$(cat /jet/home/swang47/.cache/huggingface/token)
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

EXP_NAME="${1:?Usage: sbatch ... script.sh EXP_NAME METHOD DATASET LR EPOCHS GA WARMUP WD [EXTRA_ARGS...]}"
METHOD="${2:?}"
DATASET="${3:?}"
LR="${4:?}"
EPOCHS="${5:?}"
GA="${6:?}"
WARMUP="${7:?}"
WD="${8:?}"
shift 8

PROJECT=/jet/home/swang47/yang/projects/LostInTheSecond
TRAIN_SCRIPT=$PROJECT/scripts/data_prep/15_finetune_lemma.py
OUTPUT_DIR=/ocean/projects/cis250050p/swang47/yang/LostInTheSecond/artifacts/full_prefill_llama8b/$EXP_NAME
MODEL_ID=meta-llama/Meta-Llama-3-8B

cd $PROJECT

echo "======================================================"
echo "  Training: $EXP_NAME ($METHOD)"
echo "  Dataset: $DATASET"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  Started: $(date)"
echo "======================================================"

python3 -u $TRAIN_SCRIPT \
    --method $METHOD \
    --model-id $MODEL_ID \
    --dataset-path $DATASET \
    --output-dir $OUTPUT_DIR \
    --learning-rate $LR \
    --num-train-epochs $EPOCHS \
    --gradient-accumulation-steps $GA \
    --warmup-ratio $WARMUP \
    --weight-decay $WD \
    --per-device-train-batch-size 1 \
    --logging-steps 1 \
    --save-total-limit 2 \
    --seed 42 \
    "$@"

# If LoRA, merge automatically
if [ "$METHOD" = "lora" ]; then
    ADAPTER_DIR=$OUTPUT_DIR/final_adapter
    MERGED_DIR=$OUTPUT_DIR/merged_model
    if [ -d "$ADAPTER_DIR" ] && [ ! -f "$MERGED_DIR/config.json" ]; then
        echo ""
        echo "=== Merging LoRA adapter ==="
        python3 -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
dtype = torch.bfloat16 if use_bf16 else torch.float16
print(f'Merge dtype: {dtype} (GPU: {gpu_name})')
base = AutoModelForCausalLM.from_pretrained('$MODEL_ID', torch_dtype=dtype)
model = PeftModel.from_pretrained(base, '$ADAPTER_DIR')
merged = model.merge_and_unload()
merged.save_pretrained('$MERGED_DIR')
AutoTokenizer.from_pretrained('$MODEL_ID').save_pretrained('$MERGED_DIR')
print('Merged -> $MERGED_DIR')
"
    fi
fi

echo ""
echo "======================================================"
echo "  Completed: $EXP_NAME"
echo "  Ended: $(date)"
echo "======================================================"
