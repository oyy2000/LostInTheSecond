#!/usr/bin/env python3
"""
Statistical validation of LoRA and Full FT improvements over the base model.

Reads per-sample prediction files (samples_*.jsonl) from both LoRA and
full FT evaluation directories, then computes:
  - McNemar's exact test (paired binary comparison)
  - Bootstrap confidence intervals (10,000 resamples) for EM delta
  - Bonferroni-corrected significance for top-K comparisons

Output: documents/statistical_validation.md

Usage:
    python 20_statistical_validation.py
    python 20_statistical_validation.py --top-k 10
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

BEEGFS_ARTIFACTS = Path("/mnt/beegfs/youyang7/projects/LostInSecond/artifacts")
LORA_EVAL_ROOT = BEEGFS_ARTIFACTS / "lora_sweep" / "_gsm8k_vllm_eval"
FT_EVAL_ROOT = BEEGFS_ARTIFACTS / "full_ft_sweep" / "_gsm8k_vllm_eval"

LOCAL_LORA_EVAL = PROJECT_ROOT / "artifacts" / "lora_sweep" / "_gsm8k_vllm_eval"
LOCAL_FT_EVAL = PROJECT_ROOT / "artifacts" / "full_ft_sweep" / "_gsm8k_vllm_eval"


def find_eval_root(candidate_paths: List[Path]) -> Optional[Path]:
    for p in candidate_paths:
        if p.exists():
            return p
    return None


def find_samples_file(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.rglob("samples_gsm8k_cot_zeroshot_unified_*.jsonl"))
    return candidates[-1] if candidates else None


def load_per_sample_scores(samples_path: Path) -> Dict[int, float]:
    """Load per-sample exact_match scores keyed by doc_id."""
    scores = {}
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            em = obj.get("exact_match")
            if doc_id is not None and em is not None:
                scores[doc_id] = float(em)
    return scores


def mcnemar_exact_test(base_scores: Dict[int, float],
                       model_scores: Dict[int, float]) -> Dict[str, Any]:
    """
    McNemar's exact test for paired binary outcomes.
    b = base wrong, model correct  (improvements)
    c = base correct, model wrong  (regressions)
    """
    common_ids = sorted(set(base_scores) & set(model_scores))
    n = len(common_ids)

    both_correct = both_wrong = b_count = c_count = 0
    for doc_id in common_ids:
        base_ok = base_scores[doc_id] >= 0.5
        model_ok = model_scores[doc_id] >= 0.5
        if base_ok and model_ok:
            both_correct += 1
        elif not base_ok and not model_ok:
            both_wrong += 1
        elif not base_ok and model_ok:
            b_count += 1
        else:
            c_count += 1

    total_discordant = b_count + c_count
    if total_discordant == 0:
        p_value = 1.0
    else:
        k = min(b_count, c_count)
        p_value = 0.0
        for i in range(k + 1):
            p_value += math.comb(total_discordant, i) * (0.5 ** total_discordant)
        p_value *= 2
        p_value = min(p_value, 1.0)

    return {
        "n": n,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "base_wrong_model_correct": b_count,
        "base_correct_model_wrong": c_count,
        "p_value": p_value,
    }


def bootstrap_ci(base_scores: Dict[int, float],
                  model_scores: Dict[int, float],
                  n_bootstrap: int = 10000,
                  alpha: float = 0.05,
                  seed: int = 42) -> Dict[str, float]:
    """Bootstrap confidence interval for the EM delta."""
    rng = random.Random(seed)
    common_ids = sorted(set(base_scores) & set(model_scores))
    n = len(common_ids)

    base_arr = [base_scores[i] for i in common_ids]
    model_arr = [model_scores[i] for i in common_ids]

    observed_delta = sum(model_arr) / n - sum(base_arr) / n

    deltas = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        b_mean = sum(base_arr[i] for i in indices) / n
        m_mean = sum(model_arr[i] for i in indices) / n
        deltas.append(m_mean - b_mean)

    deltas.sort()
    lo_idx = int(n_bootstrap * (alpha / 2))
    hi_idx = int(n_bootstrap * (1 - alpha / 2))

    return {
        "observed_delta": observed_delta,
        "ci_lower": deltas[lo_idx],
        "ci_upper": deltas[hi_idx],
        "bootstrap_mean": sum(deltas) / len(deltas),
        "bootstrap_std": (sum((d - sum(deltas)/len(deltas))**2 for d in deltas) / len(deltas)) ** 0.5,
    }


def discover_models(eval_root: Path,
                    method: str) -> List[Dict[str, Any]]:
    """Discover all evaluated models under an eval root."""
    models = []
    if not eval_root.exists():
        return models

    for d in sorted(eval_root.iterdir()):
        if not d.is_dir():
            continue
        samples_file = find_samples_file(d)
        if samples_file is None:
            continue
        models.append({
            "name": d.name,
            "method": method,
            "samples_path": samples_file,
        })
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10,
                    help="Number of top models to validate (for Bonferroni)")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    args = ap.parse_args()

    lora_root = find_eval_root([LORA_EVAL_ROOT, LOCAL_LORA_EVAL])
    ft_root = find_eval_root([FT_EVAL_ROOT, LOCAL_FT_EVAL])

    all_models = []
    if lora_root:
        all_models.extend(discover_models(lora_root, "lora"))
    if ft_root:
        all_models.extend(discover_models(ft_root, "full_ft"))

    base_models = [m for m in all_models if m["name"] == "base"]
    non_base = [m for m in all_models if m["name"] != "base"]

    if not base_models:
        print("ERROR: No base model predictions found.")
        return

    base = base_models[0]
    print(f"Loading base model predictions from: {base['samples_path']}")
    base_scores = load_per_sample_scores(base["samples_path"])
    base_em = sum(base_scores.values()) / len(base_scores) if base_scores else 0

    print(f"Base model: {len(base_scores)} samples, EM = {base_em:.4f}")

    # Deduplicate models by name (prefer lora root if same name in both)
    seen = set()
    unique_models = []
    for m in non_base:
        if m["name"] not in seen:
            seen.add(m["name"])
            unique_models.append(m)

    print(f"\nFound {len(unique_models)} unique models to validate.")

    results = []
    for m in unique_models:
        scores = load_per_sample_scores(m["samples_path"])
        if not scores:
            continue
        em = sum(scores.values()) / len(scores)
        mcnemar = mcnemar_exact_test(base_scores, scores)
        boot = bootstrap_ci(base_scores, scores,
                            n_bootstrap=args.n_bootstrap)

        results.append({
            "name": m["name"],
            "method": m["method"],
            "em": em,
            "delta": em - base_em,
            **mcnemar,
            **boot,
        })

    results.sort(key=lambda x: -x["em"])

    bonferroni_k = min(args.top_k, len(results))
    alpha_corrected = 0.05 / bonferroni_k if bonferroni_k > 0 else 0.05

    # Generate report
    lines = [
        "# Statistical Validation of LoRA / Full FT Results",
        "",
        f"**Base Model EM**: {base_em:.4f} ({len(base_scores)} samples)",
        f"**Bootstrap resamples**: {args.n_bootstrap}",
        f"**Bonferroni correction**: top-{bonferroni_k} comparisons, "
        f"alpha_corrected = {alpha_corrected:.4f}",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Top Models — Statistical Tests",
        "",
        "| Rank | Name | Method | EM | Delta | 95% CI | McNemar p | "
        "b (fix) | c (break) | Sig (0.05) | Sig (Bonf) |",
        "|------|------|--------|----|-------|--------|-----------|"
        "---------|-----------|------------|------------|",
    ]

    for i, r in enumerate(results[:args.top_k]):
        ci_str = f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]"
        sig_05 = "YES" if r["p_value"] < 0.05 else "no"
        sig_bonf = "YES" if r["p_value"] < alpha_corrected else "no"
        lines.append(
            f"| {i+1} | {r['name']} | {r['method']} | "
            f"{r['em']:.4f} | {r['delta']:+.4f} | {ci_str} | "
            f"{r['p_value']:.4f} | {r['base_wrong_model_correct']} | "
            f"{r['base_correct_model_wrong']} | {sig_05} | {sig_bonf} |"
        )

    lines.extend(["", "## Interpretation", ""])

    sig_count = sum(1 for r in results[:bonferroni_k] if r["p_value"] < 0.05)
    bonf_count = sum(1 for r in results[:bonferroni_k]
                     if r["p_value"] < alpha_corrected)

    lines.append(f"- {sig_count}/{bonferroni_k} models significant at p < 0.05 "
                 "(uncorrected)")
    lines.append(f"- {bonf_count}/{bonferroni_k} models significant after "
                 "Bonferroni correction")

    if results:
        best = results[0]
        lines.extend([
            "",
            f"**Best model**: `{best['name']}` "
            f"(EM = {best['em']:.4f}, delta = {best['delta']:+.4f})",
            f"- McNemar p-value: {best['p_value']:.4f}",
            f"- 95% bootstrap CI for delta: "
            f"[{best['ci_lower']:+.4f}, {best['ci_upper']:+.4f}]",
            f"- Fixed {best['base_wrong_model_correct']} problems, "
            f"broke {best['base_correct_model_wrong']} problems",
        ])

    lines.extend([
        "",
        "## All Models (Full Table)",
        "",
        "| Name | Method | EM | Delta | McNemar p | b (fix) | c (break) |",
        "|------|--------|----|-------|-----------|---------|-----------|",
    ])

    for r in results:
        lines.append(
            f"| {r['name']} | {r['method']} | {r['em']:.4f} | "
            f"{r['delta']:+.4f} | {r['p_value']:.4f} | "
            f"{r['base_wrong_model_correct']} | "
            f"{r['base_correct_model_wrong']} |"
        )

    lines.extend(["", "---",
                   f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*", ""])

    output_path = DOCUMENTS_DIR / "statistical_validation.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {output_path}")

    # Console summary
    print(f"\n{'=' * 80}")
    print(f"{'STATISTICAL VALIDATION SUMMARY':^80}")
    print(f"{'=' * 80}")
    print(f"Base EM: {base_em:.4f}")
    for i, r in enumerate(results[:5]):
        sig = "*" if r["p_value"] < 0.05 else " "
        print(f"  {i+1}. {r['name']:<28} EM={r['em']:.4f}  "
              f"delta={r['delta']:+.4f}  p={r['p_value']:.4f} {sig}")


if __name__ == "__main__":
    main()
