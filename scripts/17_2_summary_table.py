#!/usr/bin/env python3
"""Generate cross-dataset summary table for LR-boost results."""
import json
from pathlib import Path

base = Path("results/qwen2.5_3b_instruct_budget_multisignal")
datasets = ["math500", "gsm8k", "amc2023", "olympiadbench"]

# Load all results
all_data = {}
for ds in datasets:
    path = base / ds / "ac_evaluation" / "lr_boost_results.json"
    if path.exists():
        all_data[ds] = json.loads(path.read_text())
        print(f"Loaded {ds}: {len(all_data[ds])} entries")

# Print PoLR-style table
print()
print("LLM: Qwen2.5-3B-Instruct    N = 16 (drafts)")
print("Metric: G/kT = (Acc% - Greedy%) / kTokens. Higher is better.")
print()

# Header
print(f'{"Method":<35}', end='')
for ds in datasets:
    print(f' | {ds:<20}', end='')
print()
print(f'{"":<35}', end='')
for ds in datasets:
    print(f' | {"Acc":>6} {"kTok":>5} {"G/kT":>6}', end='')
print()
print('-' * 120)

# Methods to show
method_groups = [
    ("Greedy@1", lambda r: r["method"] == "Greedy@1"),
    ("SC@N", lambda r: r["method"].startswith("SC@")),
    ("", None),  # separator
    ("AC(c=0.95)", lambda r: r["method"] == "AC(c=0.95)" and r.get("strategy","") == "prm_drop_fb_last"),
    ("AC+StepBack(s=1,prm)", lambda r: "AC(c=0.95)+StepBack(s=1,prm_drop_fb_last)" in r["method"] or "AC(c=0.95)+LR(s=1,prm_drop_fb_last)" in r["method"]),
    ("AC+StepBack(s=2,prm)", lambda r: "AC(c=0.95)+StepBack(s=2,prm_drop_fb_last)" in r["method"] or "AC(c=0.95)+LR(s=2,prm_drop_fb_last)" in r["method"]),
    ("", None),  # separator
    ("ESC(w=5)", lambda r: r["method"] == "ESC(w=5)" and r.get("strategy","") == "prm_drop_fb_last"),
    ("ESC+StepBack(s=1,prm)", lambda r: "ESC(w=5)+StepBack(s=1,prm_drop_fb_last)" in r["method"] or "ESC(w=5)+LR(s=1,prm_drop_fb_last)" in r["method"]),
    ("ESC+StepBack(s=2,prm)", lambda r: "ESC(w=5)+StepBack(s=2,prm_drop_fb_last)" in r["method"] or "ESC(w=5)+LR(s=2,prm_drop_fb_last)" in r["method"]),
    ("", None),  # separator
    ("DSC(c=0.95)", lambda r: r["method"] == "DSC(c=0.95)" and r.get("strategy","") == "prm_drop_fb_last"),
    ("DSC+StepBack(s=1,prm)", lambda r: "DSC(c=0.95)+StepBack(s=1,prm_drop_fb_last)" in r["method"] or "DSC(c=0.95)+LR(s=1,prm_drop_fb_last)" in r["method"]),
    ("DSC+StepBack(s=2,prm)", lambda r: "DSC(c=0.95)+StepBack(s=2,prm_drop_fb_last)" in r["method"] or "DSC(c=0.95)+LR(s=2,prm_drop_fb_last)" in r["method"]),
]

for label, pred in method_groups:
    if pred is None:
        print()
        continue
    print(f'{label:<35}', end='')
    for ds in datasets:
        results = all_data.get(ds, [])
        matches = [r for r in results if pred(r)]
        if matches:
            r = matches[0]
            acc = r["accuracy"] * 100
            ktok = r["avg_tokens"] / 1000
            gkt = r["acc_gain_per_ktoken"] * 100
            print(f' | {acc:>6.2f} {ktok:>5.2f} {gkt:>6.3f}', end='')
        else:
            print(f' |     --    --     --', end='')
    print()

print()
