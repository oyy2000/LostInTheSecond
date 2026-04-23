#!/usr/bin/env python3
"""Debug: check MATH500 data format and Phase 0 output."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 1. Check MATH500 data format
math_path = ROOT / "lm-evaluation-harness/math_eval_data/MATH-500/test.jsonl"
lines = math_path.read_text("utf-8").splitlines()
print(f"MATH500 lines: {len(lines)}")
d = json.loads(lines[0])
print(f"Keys: {list(d.keys())}")
for k, v in d.items():
    print(f"  {k}: {str(v)[:120]}")

# 2. Check Phase 0 output
out_path = ROOT / "results/math500_3b_multi_sample/raw_cot_n8.jsonl"
if out_path.exists():
    out_lines = out_path.read_text("utf-8").splitlines()
    print(f"\nPhase 0 output lines: {len(out_lines)}")
    if out_lines:
        r = json.loads(out_lines[0])
        print(f"First record keys: {list(r.keys())}")
        for k, v in r.items():
            print(f"  {k}: {str(v)[:120]}")
else:
    print(f"\nPhase 0 output NOT FOUND: {out_path}")

# 3. Check shard files
shard_dir = ROOT / "results/math500_3b_multi_sample/_shards"
if shard_dir.exists():
    for sf in sorted(shard_dir.glob("shard_*.jsonl")):
        n = len(sf.read_text("utf-8").splitlines())
        print(f"  {sf.name}: {n} lines")
else:
    print("Shard dir not found")

# 4. Test answer equivalence
import sys
sys.path.insert(0, str(ROOT))
from src.math_answer_equiv import extract_boxed_answer, is_math_equiv

test_cases = [
    ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
    ("\\boxed{42}", "42"),
    ("The answer is \\boxed{3x+2}", "3x+2"),
]
print("\nAnswer equiv tests:")
for text, gold in test_cases:
    pred = extract_boxed_answer(text)
    eq = is_math_equiv(pred, gold)
    print(f"  extract('{text[:40]}') = '{pred}', equiv('{pred}', '{gold}') = {eq}")
