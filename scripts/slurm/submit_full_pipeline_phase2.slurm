#!/bin/bash
#SBATCH --job-name=lits-p2
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=/jet/home/swang47/yang/projects/LostInTheSecond/logs/phase2-%j.out
#SBATCH --error=/jet/home/swang47/yang/projects/LostInTheSecond/logs/phase2-%j.err

set -euo pipefail

source /ocean/projects/cis250050p/swang47/miniconda3/etc/profile.d/conda.sh
conda activate sft_yang

export PYTHONUNBUFFERED=1

cd /jet/home/swang47/yang/projects/LostInTheSecond
ARTIFACTS=./artifacts_real/full

echo "================================================"
echo "  Phase 2: GPT-5.1 Step 2 correction (CPU-only, needs network)"
echo "  Node: $(hostname)"
echo "  Started: $(date)"
echo "================================================"

GENERATIONS="$ARTIFACTS/lemma_llama3_generations.json"
FIX_FILE="$ARTIFACTS/lemma_ds2_fix_step2_gpt.json"
WAIT_FILE="$ARTIFACTS/lemma_ds2_wait_recompute_gpt.json"
AUDIT_FILE="$ARTIFACTS/lemma_gpt_fix_step2_audit.json"

if [ ! -f "$GENERATIONS" ]; then
    echo "ERROR: Phase 1 output not found: $GENERATIONS"
    exit 1
fi

echo ""
echo "=== GPT-5.1 Step 2 correction ==="
python3 scripts/data_prep/12_gpt_fix_step2_lemma.py \
    --in-file "$GENERATIONS" \
    --out-fix "$FIX_FILE" \
    --out-wait "$WAIT_FILE" \
    --audit-json "$AUDIT_FILE" \
    --model gpt-5.1 \
    --limit 0 \
    --judge-first

echo ""
echo "================================================"
echo "  Phase 2 complete!"
echo "  Fix Step2: $FIX_FILE"
echo "  Wait+Recompute: $WAIT_FILE"
echo "  Ended: $(date)"
echo "================================================"
