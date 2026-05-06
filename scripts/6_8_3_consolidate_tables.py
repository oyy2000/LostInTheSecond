#!/usr/bin/env python3
"""
Consolidate eval_summary_table.md from all models and datasets into one markdown.

Usage:
    python scripts/6_8_3_consolidate_tables.py [--out results/all_eval_tables.md]
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

MODEL_DIRS = [
    ("Qwen2.5-3B-Instruct", "qwen2.5_3b_instruct_budget_multisignal"),
    ("DeepSeek-R1-Distill-Qwen-7B", "deepseek_r1_distill_qwen_7b_budget_multisignal"),
    ("Llama-3.2-3B-Instruct", "llama_3.2_3b_instruct_budget_multisignal"),
]

SKIP_DIRS = {"math500_seed42", "math500_seed123", "math500_seed456",
             "math500_seed789", "math500_seed1024"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="",
                    help="Output path (default: results/all_eval_tables.md)")
    return ap.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out) if args.out else RESULTS_ROOT / "all_eval_tables.md"

    lines = ["# Consolidated Eval Summary Tables", ""]

    for model_name, model_dir in MODEL_DIRS:
        base = RESULTS_ROOT / model_dir
        if not base.exists():
            continue

        lines.append(f"## {model_name}")
        lines.append("")

        ds_dirs = sorted(
            d for d in base.iterdir()
            if d.is_dir() and d.name not in SKIP_DIRS
        )

        for ds_path in ds_dirs:
            table_file = ds_path / "eval_summary_table.md"
            if not table_file.exists():
                continue

            lines.append(f"### {ds_path.name}")
            lines.append("")
            lines.append(table_file.read_text().strip())
            lines.append("")

    content = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    print(f"Saved to: {out_path}")
    print(f"  ({len(lines)} lines)")


if __name__ == "__main__":
    main()