#!/usr/bin/env python3
"""
Unified comparison report for base model finetuning experiments.

Loads results from all 8 sweep directories (LoRA/FullFT x prefill/WR x 2 models)
and optionally the original instruct model results. Generates a comprehensive
comparison report at documents/base_model_comparison.md.

Usage:
    python 25_eval_compare_base_models.py
    python 25_eval_compare_base_models.py --include-instruct
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

OCEAN_ROOT = Path("/ocean/projects/cis250050p/swang47/yang/LostInTheSecond")
ARTIFACTS_ROOT = OCEAN_ROOT / "artifacts"

ARTIFACTS_LOCAL = PROJECT_ROOT / "artifacts_local"

SWEEP_DIRS = {
    # (model_short, method, dataset_label): sweep_root
    ("qwen25_3b_base", "lora", "prefill"):
        ARTIFACTS_ROOT / "lora_sweep_qwen25_3b_base",
    ("qwen25_3b_base", "lora", "wait_recompute"):
        ARTIFACTS_ROOT / "lora_sweep_qwen25_3b_base_wr",
    ("qwen25_3b_base", "full_ft", "prefill"):
        ARTIFACTS_ROOT / "full_ft_sweep_qwen25_3b_base",
    ("qwen25_3b_base", "full_ft", "wait_recompute"):
        ARTIFACTS_ROOT / "full_ft_sweep_qwen25_3b_base_wr",
    ("llama3_8b_base", "lora", "prefill"):
        ARTIFACTS_ROOT / "lora_sweep_llama3_8b_base",
    ("llama3_8b_base", "lora", "wait_recompute"):
        ARTIFACTS_ROOT / "lora_sweep_llama3_8b_base_wr",
    ("llama3_8b_base", "full_ft", "prefill"):
        ARTIFACTS_ROOT / "full_ft_sweep_llama3_8b_base",
    ("llama3_8b_base", "full_ft", "wait_recompute"):
        ARTIFACTS_ROOT / "full_ft_sweep_llama3_8b_base_wr",
}

INSTRUCT_SWEEP_DIRS = {
    ("qwen25_3b_instruct", "lora", "prefill"):
        ARTIFACTS_LOCAL / "lora_sweep",
    ("qwen25_3b_instruct", "lora", "wait_recompute"):
        ARTIFACTS_LOCAL / "lora_sweep_wr",
    ("qwen25_3b_instruct", "full_ft", "prefill"):
        ARTIFACTS_LOCAL / "full_ft_sweep",
    ("qwen25_3b_instruct", "full_ft", "wait_recompute"):
        ARTIFACTS_LOCAL / "full_ft_sweep_wr",
}

MODEL_LABELS = {
    "qwen25_3b_base": "Qwen2.5-3B (base)",
    "llama3_8b_base": "Llama-3-8B (base)",
    "qwen25_3b_instruct": "Qwen2.5-3B-Instruct",
}


def load_results(sweep_root: Path) -> Dict[str, Dict]:
    """Load results from all eval subdirectories in a sweep root."""
    results: Dict[str, Dict] = {}
    for task_short in ["gsm8k", "math500"]:
        eval_root = sweep_root / f"_{task_short}_eval"
        results_file = eval_root / "all_results.json"
        if not results_file.exists():
            # Try the older naming pattern (instruct experiments)
            eval_root = sweep_root / "_gsm8k_vllm_eval"
            results_file = eval_root / "all_results.json"
            if not results_file.exists():
                continue
            # Old format only has gsm8k
            for entry in json.loads(results_file.read_text()):
                name = entry["name"]
                if name not in results:
                    results[name] = {"name": name}
                if "gsm8k_em" in entry:
                    results[name]["gsm8k_em"] = entry["gsm8k_em"]
                elif task_short == "gsm8k":
                    results[name]["gsm8k_em"] = entry.get("gsm8k_em")
            continue

        for entry in json.loads(results_file.read_text()):
            name = entry["name"]
            if name not in results:
                results[name] = {"name": name}
            results[name].update(entry)

    return results


def get_best_and_baseline(results: Dict[str, Dict]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Return (best_adapter, baseline) from results dict."""
    baseline = results.get("base")
    adapters = [(n, r) for n, r in results.items() if n != "base" and r.get("gsm8k_em") is not None]
    if not adapters:
        return None, baseline
    best = max(adapters, key=lambda x: x[1]["gsm8k_em"])
    return best[1], baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-instruct", action="store_true",
                    help="Include original instruct model results if available")
    args = ap.parse_args()

    all_sweep_dirs = dict(SWEEP_DIRS)
    if args.include_instruct:
        all_sweep_dirs.update(INSTRUCT_SWEEP_DIRS)

    # Collect all results
    all_results: Dict[Tuple, Dict[str, Dict]] = {}
    for key, sweep_root in all_sweep_dirs.items():
        if sweep_root.exists():
            results = load_results(sweep_root)
            if results:
                all_results[key] = results
                print(f"  Loaded {len(results)} entries from {sweep_root.name}")
            else:
                print(f"  No results in {sweep_root.name}")
        else:
            print(f"  Not found: {sweep_root}")

    if not all_results:
        print("No results found. Run experiments first.")
        return

    # Build report
    lines = [
        "# Base Model Finetuning -- Comparison Report",
        "",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. Baseline Performance (No Finetuning)",
        "",
        "| Model | GSM8K EM | MATH-500 EM |",
        "|-------|----------|-------------|",
    ]

    seen_baselines = {}
    for (model_short, method, ds), results in all_results.items():
        baseline = results.get("base", {})
        if model_short not in seen_baselines and baseline:
            seen_baselines[model_short] = baseline
            label = MODEL_LABELS.get(model_short, model_short)
            gsm = baseline.get("gsm8k_em")
            math = baseline.get("math500_em")
            gsm_str = f"{gsm:.4f}" if gsm is not None else "N/A"
            math_str = f"{math:.4f}" if math is not None else "N/A"
            lines.append(f"| {label} | {gsm_str} | {math_str} |")

    # Best per configuration
    lines.extend([
        "",
        "## 2. Best Finetuned Model Per Configuration",
        "",
        "| Model | Method | Dataset | Best Config | GSM8K EM | GSM8K Delta "
        "| MATH-500 EM | MATH Delta |",
        "|-------|--------|---------|-------------|----------|------------|"
        "-------------|------------|",
    ])

    summary_rows: List[Dict] = []
    for (model_short, method, ds), results in sorted(all_results.items()):
        best, baseline = get_best_and_baseline(results)
        if best is None:
            continue

        label = MODEL_LABELS.get(model_short, model_short)
        base_gsm = baseline.get("gsm8k_em") if baseline else None
        base_math = baseline.get("math500_em") if baseline else None

        gsm = best.get("gsm8k_em")
        math = best.get("math500_em")
        gsm_str = f"{gsm:.4f}" if gsm is not None else "N/A"
        math_str = f"{math:.4f}" if math is not None else "N/A"
        gsm_d = f"{gsm - base_gsm:+.4f}" if (gsm is not None and base_gsm is not None) else "N/A"
        math_d = f"{math - base_math:+.4f}" if (math is not None and base_math is not None) else "N/A"
        ds_label = "W+R" if ds == "wait_recompute" else "Prefill"

        lines.append(
            f"| {label} | {method} | {ds_label} | {best['name']} "
            f"| {gsm_str} | {gsm_d} | {math_str} | {math_d} |"
        )

        summary_rows.append({
            "model": model_short, "method": method, "dataset": ds,
            "best_name": best["name"],
            "gsm8k_em": gsm, "math500_em": math,
            "gsm8k_delta": (gsm - base_gsm) if (gsm and base_gsm) else None,
            "math500_delta": (math - base_math) if (math and base_math) else None,
        })

    # Method comparison
    lines.extend([
        "",
        "## 3. LoRA vs Full FT Comparison",
        "",
        "| Model | Dataset | LoRA Best GSM8K | Full FT Best GSM8K "
        "| LoRA Best MATH | Full FT Best MATH |",
        "|-------|---------|-----------------|--------------------"
        "|----------------|-------------------|",
    ])

    for model_short in ["qwen25_3b_base", "llama3_8b_base"]:
        for ds in ["prefill", "wait_recompute"]:
            label = MODEL_LABELS.get(model_short, model_short)
            ds_label = "W+R" if ds == "wait_recompute" else "Prefill"

            lora_rows = [r for r in summary_rows
                         if r["model"] == model_short and r["method"] == "lora"
                         and r["dataset"] == ds]
            ft_rows = [r for r in summary_rows
                       if r["model"] == model_short and r["method"] == "full_ft"
                       and r["dataset"] == ds]

            lora_gsm = f"{lora_rows[0]['gsm8k_em']:.4f}" if lora_rows and lora_rows[0].get("gsm8k_em") else "N/A"
            ft_gsm = f"{ft_rows[0]['gsm8k_em']:.4f}" if ft_rows and ft_rows[0].get("gsm8k_em") else "N/A"
            lora_math = f"{lora_rows[0]['math500_em']:.4f}" if lora_rows and lora_rows[0].get("math500_em") else "N/A"
            ft_math = f"{ft_rows[0]['math500_em']:.4f}" if ft_rows and ft_rows[0].get("math500_em") else "N/A"

            lines.append(
                f"| {label} | {ds_label} | {lora_gsm} | {ft_gsm} "
                f"| {lora_math} | {ft_math} |"
            )

    # Detailed results per sweep
    lines.extend([
        "",
        "## 4. Detailed Results Per Sweep",
        "",
    ])

    for (model_short, method, ds), results in sorted(all_results.items()):
        label = MODEL_LABELS.get(model_short, model_short)
        ds_label = "W+R" if ds == "wait_recompute" else "Prefill"
        baseline = results.get("base", {})
        base_gsm = baseline.get("gsm8k_em")

        adapters = sorted(
            [(n, r) for n, r in results.items()
             if n != "base" and r.get("gsm8k_em") is not None],
            key=lambda x: -(x[1].get("gsm8k_em") or -1)
        )

        if not adapters:
            continue

        lines.extend([
            f"### {label} / {method} / {ds_label}",
            "",
            f"Baseline GSM8K EM: **{base_gsm}**",
            "",
            "| Rank | Name | GSM8K EM | Delta | MATH-500 EM |",
            "|------|------|----------|-------|-------------|",
        ])

        for i, (name, r) in enumerate(adapters[:10]):
            gsm = r.get("gsm8k_em")
            math = r.get("math500_em")
            gsm_str = f"{gsm:.4f}" if gsm is not None else "FAIL"
            math_str = f"{math:.4f}" if math is not None else "N/A"
            d = f"{gsm - base_gsm:+.4f}" if (gsm is not None and base_gsm is not None) else "N/A"
            lines.append(f"| {i+1} | {name} | {gsm_str} | {d} | {math_str} |")

        if len(adapters) > 10:
            lines.append(f"| ... | ({len(adapters) - 10} more) | ... | ... | ... |")

        lines.append("")

    lines.extend([
        "---", "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*", "",
    ])

    output_path = DOCUMENTS_DIR / "base_model_comparison.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {output_path}")

    # Also save raw summary JSON
    summary_json = ARTIFACTS_ROOT / "comparison_summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary_rows, indent=2, default=str))
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
