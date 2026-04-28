#!/usr/bin/env python3
"""Launcher: runs 7_1, 7_2, 7_3 in sequence. Logs to logs/7_run_all.log."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
os.makedirs(ROOT / "logs", exist_ok=True)

PYTHON = "/common/users/sl2148/anaconda3/envs/rl_steer_yang/bin/python3"

log_path = ROOT / "logs" / "7_run_all.log"
log_f = open(log_path, "w")

def run(cmd_str):
    print(f">>> {cmd_str}", flush=True)
    log_f.write(f">>> {cmd_str}\n")
    log_f.flush()
    proc = subprocess.run(
        cmd_str, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    log_f.write(proc.stdout)
    log_f.write(f"[exit code: {proc.returncode}]\n\n")
    log_f.flush()
    print(proc.stdout[-500:] if len(proc.stdout) > 500 else proc.stdout, flush=True)
    if proc.returncode != 0:
        print(f"FAILED with exit code {proc.returncode}", flush=True)
        sys.exit(1)

print("=== Late Rollback Experiment Launcher ===", flush=True)

run(f"{PYTHON} scripts/7_1_late_rollback_suffix_vote.py "
    f"--alpha 0.6,0.7,0.8 --K 8 --gpus 2,3,4,5,6,7")

run(f"{PYTHON} scripts/7_2_full_self_consistency.py "
    f"--K 8 --gpus 2,3,4,5,6,7")

run(f"{PYTHON} scripts/7_3_compare_late_rollback_vs_sc.py")

log_f.close()
print("=== ALL DONE ===", flush=True)
