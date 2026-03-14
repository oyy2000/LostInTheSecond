#!/usr/bin/env python3
"""Fix all_results.json to use latest flexible-extract metric."""
import json
from pathlib import Path

EVAL_ROOT = Path("artifacts_local/lora_sweep/_gsm8k_vllm_eval")
TASK = "gsm8k_cot_zeroshot_unified"

all_results = json.loads((EVAL_ROOT / "all_results.json").read_text())

for entry in all_results:
    name = entry["name"]
    if name == "base":
        continue
    eval_dir = EVAL_ROOT / name
    if not eval_dir.exists():
        continue
    result_files = sorted(eval_dir.rglob("results_*.json"))
    if not result_files:
        continue
    latest = result_files[-1]
    obj = json.loads(latest.read_text())
    task_block = (obj.get("results") or {}).get(TASK, {})
    for key in ["exact_match,flexible-extract", "exact_match,none",
                 "exact_match,strict-match", "exact_match"]:
        if key in task_block:
            try:
                entry["gsm8k_em"] = float(task_block[key])
                print(f"  {name}: {key} = {entry['gsm8k_em']:.4f}  (from {latest.name})")
                break
            except Exception:
                pass

(EVAL_ROOT / "all_results.json").write_text(
    json.dumps(all_results, indent=2, default=str))

base_em = next((e["gsm8k_em"] for e in all_results if e["name"] == "base"), 0)
print(f"\nBase: {base_em:.4f}\n")

results = [(e["name"], e.get("gsm8k_em")) for e in all_results if e["name"] != "base"]
results.sort(key=lambda x: -(x[1] or -1))
for name, em in results:
    if em is not None:
        delta = em - base_em
        print(f"  {name}: EM={em:.4f}  delta={delta:+.4f}")
    else:
        print(f"  {name}: MISSING")
