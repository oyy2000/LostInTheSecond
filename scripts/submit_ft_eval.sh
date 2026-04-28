#!/bin/bash
{
  echo "=== Submitting FT combined11k eval ==="
  sbatch /jet/home/swang47/yang/projects/LostInTheSecond/scripts/slurm/30_lemma_eval_gsm8k.sh \
    ft_combined11k \
    /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/artifacts/full_prefill_llama8b/ft_combined11k/best_model
  echo ""
  echo "=== Current queue ==="
  squeue -u swang47 --format="%.18i %.30j %.8T %.10M %.20R"
  echo ""
  echo "=== Done ==="
} > /jet/home/swang47/yang/projects/LostInTheSecond/logs/submit_ft_eval_output.txt 2>&1
