#!/usr/bin/env python3
"""
Summarize rollback fair results across all datasets and models.

Collects rollback_prm_drop_fb_last_nd16_ns2_fair and
rollback_nll_drop_fb_last_nd16_ns2_fair from each eval_summary.json,
along with SC@32 and Greedy@1 baselines.

Usage:
    python scripts/6_8_2_summarize_fair.py [--out results/fair_summary.md]
"""

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

MODEL_DIRS = [
    "qwen2.5_3b_instruct_budget_multisignal",
    "deepseek_r1_distill_qwen_7b_budget_multisignal",
    "llama_3.2_3b_instruct_budget_multisignal",
]

MODEL_SHORT = {
    "qwen2.5_3b_instruct_budget_multisignal": "Qwen2.5-3B",
    "deepseek_r1_distill_qwen_7b_budget_multisignal": "DeepSeek-R1-7B",
    "llama_3.2_3b_instruct_budget_multisignal": "Llama-3.2-3B",
}

METHODS_OF_INTEREST = [
    "Greedy@1",
    "SC@32",
    "rollback_nll_drop_fb_last_nd16_ns2",
    "rollback_nll_drop_fb_last_nd16_ns2_fair",
    "rollback_prm_drop_fb_last_nd16_ns2",
    "rollback_prm_drop_fb_last_nd16_ns2_fair",
]

METHOD_SHORT = {
    "Greedy@1": "Greedy@1",
    "SC@32": "SC@32",
    "rollback_nll_drop_fb_last_nd16_ns2": "NLL-full",
    "rollback_nll_drop_fb_last_nd16_ns2_fair": "NLL-fair",
    "rollback_prm_drop_fb_last_nd16_ns2": "PRM-full",
    "rollback_prm_drop_fb_last_nd16_ns2_fair": "PRM-fair",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="",
                    help="Output path (default: results/fair_summary.md)")
    return ap.parse_args()


def load_results(model_dir):
    """Load all eval_summary.json for a model, return {dataset: [results]}."""
    base = RESULTS_ROOT / model_dir
    if not base.exists():
        return {}
    out = {}
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        summary = d / "eval_summary.json"
        if not summary.exists():
            continue
        data = json.loads(summary.read_text())
        out[d.name] = {r["method"]: r for r in data}
    return out


def main():
    args = parse_args()
    out_path = Path(args.out) if args.out else RESULTS_ROOT / "fair_summary.md"

    lines = ["# Fair Variant Summary (nd16_ns2)", ""]

    for model_dir in MODEL_DIRS:
        model_name = MODEL_SHORT.get(model_dir, model_dir)
        datasets = load_results(model_dir)
        if not datasets:
            continue

        lines.append(f"## {model_name}")
        lines.append("")

        header = "| dataset | " + " | ".join(
            f"{METHOD_SHORT[m]} acc" for m in METHODS_OF_INTEREST
        ) + " | SC@32 tok/q | PRM-fair tok/q | PRM-fair savings |"
        sep = "|---|" + "|".join("---:" for _ in METHODS_OF_INTEREST) + "|---:|---:|---:|"
        lines.append(header)
        lines.append(sep)

        for ds_name in sorted(datasets.keys()):
            if ds_name.startswith("math500_seed"):
                continue
            by_method = datasets[ds_name]

            accs = []
            for m in METHODS_OF_INTEREST:
                r = by_method.get(m)
                if r:
                    accs.append(f"{r['acc']:.4f}")
                else:
                    accs.append("-")

            sc32 = by_method.get("SC@32")
            prm_fair = by_method.get("rollback_prm_drop_fb_last_nd16_ns2_fair")

            sc32_tpq = f"{sc32['tokens_per_q']:.0f}" if sc32 else "-"
            prm_tpq = f"{prm_fair['tokens_per_q']:.0f}" if prm_fair else "-"

            if sc32 and prm_fair:
                savings = (1 - prm_fair["tokens_per_q"] / sc32["tokens_per_q"]) * 100
                savings_str = f"{savings:.1f}%"
            else:
                savings_str = "-"

            row = f"| {ds_name} | " + " | ".join(accs) + f" | {sc32_tpq} | {prm_tpq} | {savings_str} |"
            lines.append(row)

        lines.append("")

    content = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    print(content)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
