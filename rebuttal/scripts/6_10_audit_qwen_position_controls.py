#!/usr/bin/env python3
"""Audit a completed Qwen position-control result directory."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from rebuttal.src.position_control_audit import (  # noqa: E402
    audit_position_control_results,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=500)
    parser.add_argument(
        "--expected-model",
        default="Qwen/Qwen2.5-3B-Instruct",
    )
    parser.add_argument("--expected-run-seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    report = audit_position_control_results(
        args.result_dir,
        expected_questions=args.expected_questions,
        expected_model=args.expected_model,
        expected_run_seed=args.expected_run_seed,
    )
    report_path = args.result_dir / "audit_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
