#!/usr/bin/env python3
"""
MATH-500 entrypoint for budget-controlled rollback vs self-consistency.

This wrapper runs scripts/6_6_budget_controlled_rollback.py with dataset-specific
output locations so MATH-500 results, figures, and checkpoint files do not
overwrite GSM8K outputs.

Default outputs:
  results/math500_budget_controlled/checkpoint.jsonl
  results/math500_budget_controlled/eval_summary.json
  results/math500_budget_controlled/eval_summary_table.md
  figures/budget_controlled_math500/*.pdf|png

Usage:
    systemd-run --user --scope python scripts/6_7_budget_controlled_rollback_math500.py \
        --budget 32 \
        --gpus 0,1

Small test:
    systemd-run --user --scope python scripts/6_7_budget_controlled_rollback_math500.py \
        --budget 32 \
        --n-sample 50 \
        --gpus 0,1
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "6_6_budget_controlled_rollback.py"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "math500_budget_controlled"


def main() -> None:
    cmd = [
        sys.executable,
        str(BASE_SCRIPT),
        "--dataset", "math500",
        "--out-dir", str(DEFAULT_OUT_DIR),
        "--tag", "math500",
    ] + sys.argv[1:]
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
