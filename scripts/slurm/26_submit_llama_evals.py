#!/usr/bin/env python3
"""
Submit individual 1-GPU SLURM jobs for every Llama 3 8B eval.

For each (sweep_dir, model_name, task) tuple, generates a small sbatch script
and submits it. LoRA adapters use pre-merged models; Full FT uses best_model.
Base model eval is shared across LoRA and FT within the same dataset variant.

Usage:
    python 26_submit_llama_evals.py                # submit all
    python 26_submit_llama_evals.py --dry-run      # print without submitting
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
PYTHON = "/ocean/projects/cis250050p/swang47/miniconda3/envs/sft_yang/bin/python"
PROJECT_ROOT = "/jet/home/swang47/yang/projects/LostInTheSecond"
ARTIFACTS = Path("/ocean/projects/cis250050p/swang47/yang/LostInTheSecond/artifacts")
LOG_DIR = Path("/ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs")

VLLM_ARGS = (
    "dtype=float16,"
    "gpu_memory_utilization=0.9,"
    "max_model_len=2048,"
    "max_num_seqs=16,"
    "enforce_eager=True"
)

TASKS = {
    "gsm8k": "gsm8k_cot_zeroshot_unified",
    "math500": "hendrycks_math_500",
}

SWEEP_DIRS = {
    "lora_pf":  ARTIFACTS / "lora_sweep_llama3_8b_base",
    "lora_wr":  ARTIFACTS / "lora_sweep_llama3_8b_base_wr",
    "ft_pf":    ARTIFACTS / "full_ft_sweep_llama3_8b_base",
    "ft_wr":    ARTIFACTS / "full_ft_sweep_llama3_8b_base_wr",
}

SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH -t 01:30:00
#SBATCH -J {job_name}
#SBATCH -o {log_dir}/{job_name}_%j.out
#SBATCH -e {log_dir}/{job_name}_%j.err

set -euo pipefail

export CUDA_HOME=/opt/packages/cuda/v12.1.1
export PATH="${{CUDA_HOME}}/bin:${{PATH}}"
export LD_LIBRARY_PATH="${{CUDA_HOME}}/lib64:${{LD_LIBRARY_PATH:-}}"
export PYTHONUNBUFFERED=1

echo "Job ${{SLURM_JOB_ID}} | $(hostname) | $(date)"
echo "Task: {description}"

{merge_cmd}

{python} -m lm_eval \\
    --model vllm \\
    --model_args "pretrained={model_path},{vllm_args}" \\
    --tasks {task_full} \\
    --batch_size auto \\
    --gen_kwargs "max_gen_toks=2048,temperature=0,do_sample=False" \\
    --output_path {output_path} \\
    --log_samples

echo "Done at $(date)"
"""

MERGE_SCRIPT_TEMPLATE = """\
# Merge LoRA adapter if not already merged
{python} -c "
import torch
from pathlib import Path
merged = Path('{merged_path}')
if (merged / 'config.json').exists():
    print('Already merged, skipping')
else:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained('{model_id}', torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, '{adapter_path}')
    m = model.merge_and_unload()
    merged.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(str(merged))
    AutoTokenizer.from_pretrained('{model_id}').save_pretrained(str(merged))
    print('Merged to', merged)
"
"""


def discover_eval_jobs():
    """Return list of (job_name, model_path, output_path, task_short, task_full, description, merge_cmd)."""
    jobs = []
    base_done = {}

    for sweep_key, sweep_dir in SWEEP_DIRS.items():
        if not sweep_dir.exists():
            print(f"WARN: {sweep_dir} does not exist, skipping")
            continue

        is_lora = sweep_key.startswith("lora")
        ds_tag = "pf" if "pf" in sweep_key else "wr"

        for task_short, task_full in TASKS.items():
            eval_root = sweep_dir / f"_{task_short}_eval"

            # Base model eval (shared per dataset variant + task)
            base_key = f"base_{ds_tag}_{task_short}"
            if base_key not in base_done:
                base_output = eval_root / "base"
                jname = f"ev_base_{ds_tag}_{task_short}"
                jobs.append({
                    "job_name": jname,
                    "model_path": MODEL_ID,
                    "output_path": str(base_output),
                    "task_short": task_short,
                    "task_full": task_full,
                    "description": f"Base Llama-3-8B {ds_tag} {task_short}",
                    "merge_cmd": "",
                })
                base_done[base_key] = True

            if is_lora:
                for adapter_dir in sorted(sweep_dir.glob("v3_*")):
                    if not (adapter_dir / "final_adapter").exists():
                        continue
                    name = adapter_dir.name
                    merged_path = adapter_dir / "merged_model"
                    adapter_path = adapter_dir / "final_adapter"
                    output = eval_root / name

                    merge_cmd = MERGE_SCRIPT_TEMPLATE.format(
                        python=PYTHON,
                        merged_path=merged_path,
                        model_id=MODEL_ID,
                        adapter_path=adapter_path,
                    )

                    jname = f"ev_{ds_tag}_{task_short}_{name}"[:30]
                    jobs.append({
                        "job_name": jname,
                        "model_path": str(merged_path),
                        "output_path": str(output),
                        "task_short": task_short,
                        "task_full": task_full,
                        "description": f"LoRA {name} {ds_tag} {task_short}",
                        "merge_cmd": merge_cmd,
                    })
            else:
                for exp_dir in sorted(sweep_dir.glob("ft2_*")):
                    best = exp_dir / "best_model"
                    if not best.exists():
                        continue
                    name = exp_dir.name
                    output = eval_root / name
                    jname = f"ev_{ds_tag}_{task_short}_{name}"[:30]
                    jobs.append({
                        "job_name": jname,
                        "model_path": str(best),
                        "output_path": str(output),
                        "task_short": task_short,
                        "task_full": task_full,
                        "description": f"FT {name} {ds_tag} {task_short}",
                        "merge_cmd": "",
                    })

    return jobs


def submit_job(job, dry_run=False):
    script = SBATCH_TEMPLATE.format(
        job_name=job["job_name"],
        log_dir=LOG_DIR,
        description=job["description"],
        merge_cmd=job["merge_cmd"],
        python=PYTHON,
        model_path=job["model_path"],
        vllm_args=VLLM_ARGS,
        task_full=job["task_full"],
        output_path=job["output_path"],
    )

    if dry_run:
        print(f"  [DRY-RUN] {job['job_name']}: {job['description']}")
        return None

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        f.flush()
        result = subprocess.run(
            ["sbatch", f.name],
            capture_output=True, text=True,
        )
        os.unlink(f.name)

    if result.returncode != 0:
        print(f"  FAILED {job['job_name']}: {result.stderr.strip()}")
        return None

    job_id = result.stdout.strip().split()[-1]
    print(f"  {job_id} | {job['job_name']}: {job['description']}")
    return job_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = discover_eval_jobs()

    print(f"Total eval jobs to submit: {len(jobs)}")
    if args.dry_run:
        print("\n--- DRY RUN ---")

    submitted = []
    for job in jobs:
        jid = submit_job(job, dry_run=args.dry_run)
        if jid:
            submitted.append(jid)

    if not args.dry_run:
        print(f"\nSubmitted {len(submitted)} jobs")
        print(f"Monitor: squeue -u $USER | grep ev_")
    else:
        print(f"\nWould submit {len(jobs)} jobs")


if __name__ == "__main__":
    main()
