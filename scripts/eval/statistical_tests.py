#!/usr/bin/env python3
"""
Statistical tests for the prefix correction experiment (v2).

Tests for both corrected (improvement) and corrupted (degradation) conditions.
Includes symmetry test comparing the two effects.

Tests:
  1. McNemar test: paired binary outcomes
  2. Bootstrap CI: Delta_Acc with 10000 resamples
  3. Symmetry test: |Delta_corrected| vs |Delta_corrupted|

Usage:
    python scripts/eval/statistical_tests.py \
        --cot-file results/gsm8k_7b_v2/raw_cot_n8.jsonl \
        --correction-dir results/gsm8k_7b_v2 \
        --out-dir results/gsm8k_7b_v2/statistics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.prm.scoring import split_steps

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Statistical tests (v2)")
    ap.add_argument("--cot-file",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2/raw_cot_n8.jsonl"))
    ap.add_argument("--correction-dir",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2"))
    ap.add_argument("--out-dir",
                    default=str(PROJECT_ROOT / "results/gsm8k_7b_v2/statistics"))
    ap.add_argument("--k-values", default="1,2,3,4")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


# ---------------------------------------------------------------------------
# McNemar test
# ---------------------------------------------------------------------------

def mcnemar_test(y_orig: np.ndarray, y_modified: np.ndarray) -> Dict[str, Any]:
    from scipy.stats import chi2
    b = int(np.sum((y_orig == 0) & (y_modified == 1)))
    c = int(np.sum((y_orig == 1) & (y_modified == 0)))
    n = b + c
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c, "n_discordant": n}
    stat = (abs(b - c) - 1) ** 2 / n
    p_value = float(1 - chi2.cdf(stat, df=1))
    return {
        "statistic": float(stat), "p_value": p_value,
        "b_orig_wrong_mod_right": b, "c_orig_right_mod_wrong": c,
        "n_discordant": n,
    }


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_delta_acc(
    y_orig: np.ndarray, y_modified: np.ndarray,
    n_bootstrap: int = 10000, seed: int = 42, alpha: float = 0.05,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    n = len(y_orig)
    observed_delta = float(y_modified.mean() - y_orig.mean())
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        deltas.append(y_modified[idx].mean() - y_orig[idx].mean())
    deltas = np.array(deltas)
    ci_lo = float(np.percentile(deltas, 100 * alpha / 2))
    ci_hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    if observed_delta > 0:
        p_value = float(np.mean(deltas <= 0))
    else:
        p_value = float(np.mean(deltas >= 0))
    return {
        "observed_delta_acc": observed_delta,
        "ci_lower": ci_lo, "ci_upper": ci_hi,
        "p_value_one_sided": p_value,
        "n_bootstrap": n_bootstrap,
        "acc_original": float(y_orig.mean()),
        "acc_modified": float(y_modified.mean()),
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# Symmetry test
# ---------------------------------------------------------------------------

def symmetry_test(
    delta_corrected: float, delta_corrupted: float,
    boot_corrected: np.ndarray, boot_corrupted: np.ndarray,
) -> Dict[str, Any]:
    """Test whether |Delta_corrected| == |Delta_corrupted|."""
    observed_asymmetry = abs(delta_corrected) - abs(delta_corrupted)
    boot_asymmetry = np.abs(boot_corrected) - np.abs(boot_corrupted)
    p_value = float(np.mean(boot_asymmetry <= 0)) if observed_asymmetry > 0 else \
              float(np.mean(boot_asymmetry >= 0))
    return {
        "observed_asymmetry": observed_asymmetry,
        "abs_delta_corrected": abs(delta_corrected),
        "abs_delta_corrupted": abs(delta_corrupted),
        "p_value": p_value,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    correction_dir = Path(args.correction_dir)
    k_values = [int(x) for x in args.k_values.split(",")]

    # Load original CoT
    cot_rows = read_jsonl(Path(args.cot_file))
    wrong_by_key = {}
    correct_by_key = {}
    for r in cot_rows:
        key = (r["doc_id"], r.get("sample_idx", 0))
        if r.get("exact_match", 1.0) < 1.0:
            wrong_by_key[key] = r
        else:
            correct_by_key[key] = r

    all_results = {"mcnemar": {}, "bootstrap": {}, "symmetry": {}}

    for k in k_values:
        print(f"\n=== k={k} ===")

        # --- Corrected: wrong → fixed ---
        corrected_rows = read_jsonl(correction_dir / f"prefilled_corrected_k{k}.jsonl")
        corrected_by_key = {(r["doc_id"], r.get("sample_idx", 0)): r for r in corrected_rows}

        paired_keys_corr = sorted(set(wrong_by_key.keys()) & set(corrected_by_key.keys()))
        if paired_keys_corr:
            y_orig_corr = np.array([wrong_by_key[k_]["exact_match"] for k_ in paired_keys_corr])
            y_corr = np.array([corrected_by_key[k_]["exact_match"] for k_ in paired_keys_corr])
            y_orig_corr = (y_orig_corr >= 1.0).astype(float)
            y_corr = (y_corr >= 1.0).astype(float)

            mcn_corr = mcnemar_test(y_orig_corr, y_corr)
            bs_corr = bootstrap_delta_acc(y_orig_corr, y_corr, args.n_bootstrap, args.seed)

            all_results["mcnemar"][f"corrected_k{k}"] = mcn_corr
            all_results["bootstrap"][f"corrected_k{k}"] = bs_corr
            print(f"  Corrected: n={len(paired_keys_corr)}, "
                  f"delta={bs_corr['observed_delta_acc']:.4f}, p={mcn_corr['p_value']:.4f}")
        else:
            print(f"  Corrected: no paired data")

        # --- Corrupted: correct → broken ---
        corrupted_rows = read_jsonl(correction_dir / f"prefilled_corrupted_k{k}.jsonl")
        corrupted_by_key = {(r["doc_id"], r.get("sample_idx", 0)): r for r in corrupted_rows}

        paired_keys_corrupt = sorted(set(correct_by_key.keys()) & set(corrupted_by_key.keys()))
        if paired_keys_corrupt:
            y_orig_corrupt = np.array([correct_by_key[k_]["exact_match"] for k_ in paired_keys_corrupt])
            y_corrupt = np.array([corrupted_by_key[k_]["exact_match"] for k_ in paired_keys_corrupt])
            y_orig_corrupt = (y_orig_corrupt >= 1.0).astype(float)
            y_corrupt = (y_corrupt >= 1.0).astype(float)

            mcn_corrupt = mcnemar_test(y_orig_corrupt, y_corrupt)
            bs_corrupt = bootstrap_delta_acc(y_orig_corrupt, y_corrupt, args.n_bootstrap, args.seed)

            all_results["mcnemar"][f"corrupted_k{k}"] = mcn_corrupt
            all_results["bootstrap"][f"corrupted_k{k}"] = bs_corrupt
            print(f"  Corrupted: n={len(paired_keys_corrupt)}, "
                  f"delta={bs_corrupt['observed_delta_acc']:.4f}, p={mcn_corrupt['p_value']:.4f}")
        else:
            print(f"  Corrupted: no paired data")

        # --- Symmetry test ---
        if paired_keys_corr and paired_keys_corrupt:
            rng = np.random.RandomState(args.seed)
            n_corr = len(paired_keys_corr)
            n_corrupt = len(paired_keys_corrupt)
            boot_deltas_corr = []
            boot_deltas_corrupt = []
            for _ in range(args.n_bootstrap):
                idx_c = rng.randint(0, n_corr, size=n_corr)
                boot_deltas_corr.append(y_corr[idx_c].mean() - y_orig_corr[idx_c].mean())
                idx_d = rng.randint(0, n_corrupt, size=n_corrupt)
                boot_deltas_corrupt.append(y_corrupt[idx_d].mean() - y_orig_corrupt[idx_d].mean())

            sym = symmetry_test(
                bs_corr["observed_delta_acc"],
                bs_corrupt["observed_delta_acc"],
                np.array(boot_deltas_corr),
                np.array(boot_deltas_corrupt),
            )
            all_results["symmetry"][k] = sym
            print(f"  Symmetry: |corr|={sym['abs_delta_corrected']:.4f}, "
                  f"|corrupt|={sym['abs_delta_corrupted']:.4f}, p={sym['p_value']:.4f}")

    # --- Mixed-effects GEE regression ---
    print(f"\n=== GEE Logistic Regression ===")
    try:
        import pandas as pd
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable

        # Build long-format dataframe: corrected + control only (k=0 has perfect separation)
        rows_for_reg = []

        # Corrected trajectories
        for k in k_values:
            corr_file = correction_dir / f"prefilled_corrected_k{k}.jsonl"
            if not corr_file.exists():
                continue
            for item in read_jsonl(corr_file):
                steps = item.get("all_steps", item.get("steps", []))
                if not steps:
                    steps = split_steps(item.get("full_response", item.get("response", "")),
                                        mode="double_newline")
                n_steps = len(steps)
                avg_tokens = np.mean([len(s.split()) for s in steps]) if steps else 0
                rows_for_reg.append({
                    "doc_id": item["doc_id"],
                    "sample_idx": item.get("sample_idx", 0),
                    "k": k,
                    "is_control": 0,
                    "exact_match": float(item.get("exact_match", 0.0) >= 1.0),
                    "n_steps": n_steps,
                    "avg_step_tokens": avg_tokens,
                })

        # Control trajectories
        for k in k_values:
            ctrl_file = correction_dir / f"prefilled_corrupted_k{k}.jsonl"
            if not ctrl_file.exists():
                continue
            for item in read_jsonl(ctrl_file):
                steps = item.get("all_steps", item.get("steps", []))
                if not steps:
                    steps = split_steps(item.get("full_response", item.get("response", "")),
                                        mode="double_newline")
                n_steps = len(steps)
                avg_tokens = np.mean([len(s.split()) for s in steps]) if steps else 0
                rows_for_reg.append({
                    "doc_id": item["doc_id"],
                    "sample_idx": item.get("sample_idx", 0),
                    "k": k,
                    "is_control": 1,
                    "exact_match": float(item.get("exact_match", 0.0) >= 1.0),
                    "n_steps": n_steps,
                    "avg_step_tokens": avg_tokens,
                })

        df = pd.DataFrame(rows_for_reg)
        print(f"  Regression data: {len(df)} rows, {df['doc_id'].nunique()} unique docs")
        print(f"  k distribution: {df.groupby(['is_control','k'])['exact_match'].mean().to_dict()}")

        # Predictors: k (continuous), is_control, interaction, n_steps, avg_step_tokens
        for col in ["n_steps", "avg_step_tokens"]:
            mu, sd = df[col].mean(), df[col].std()
            if sd > 0:
                df[col] = (df[col] - mu) / sd

        df["k_x_control"] = df["k"] * df["is_control"]

        X = df[["k", "is_control", "k_x_control", "n_steps", "avg_step_tokens"]].copy()
        X.insert(0, "Intercept", 1.0)
        y = df["exact_match"]
        groups = df["doc_id"]

        model = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Exchangeable())
        result = model.fit(maxiter=100)
        print(result.summary())

        reg_results = {}
        for var in X.columns:
            reg_results[var] = {
                "coef": float(result.params[var]),
                "se": float(result.bse[var]),
                "z": float(result.tvalues[var]),
                "p": float(result.pvalues[var]),
                "ci_lower": float(result.conf_int().loc[var, 0]),
                "ci_upper": float(result.conf_int().loc[var, 1]),
            }
        all_results["regression"] = reg_results
        print(f"  Regression done: {len(reg_results)} coefficients")

    except Exception as e:
        print(f"  Regression failed: {e}")
        import traceback
        traceback.print_exc()
        all_results["regression"] = {"error": str(e)}

    # Save
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_file = out_dir / "statistical_results.json"
    results_file.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=convert),
        encoding="utf-8",
    )
    print(f"\nResults -> {results_file}")

    # Summary table
    _print_summary_table(all_results, k_values)


def _print_summary_table(results: Dict, k_values: List[int]) -> None:
    print("\n" + "=" * 90)
    print(f"{'Condition':>15} | {'k':>3} | {'Acc_orig':>8} | {'Acc_mod':>8} | "
          f"{'Delta':>8} | {'95% CI':>18} | {'McNemar p':>10}")
    print("-" * 90)

    for k in k_values:
        for prefix in ["corrected", "corrupted"]:
            key = f"{prefix}_k{k}"
            bs = results["bootstrap"].get(key)
            mcn = results["mcnemar"].get(key)
            if bs is None:
                continue
            ci = f"[{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]"
            mp = f"{mcn['p_value']:.4f}" if mcn else "N/A"
            print(f"{key:>15} | {k:>3} | {bs['acc_original']:>8.4f} | "
                  f"{bs['acc_modified']:>8.4f} | {bs['observed_delta_acc']:>8.4f} | "
                  f"{ci:>18} | {mp:>10}")

    print("=" * 90)

    if results.get("symmetry"):
        print("\nSymmetry tests:")
        for k in k_values:
            sym = results["symmetry"].get(k)
            if sym:
                print(f"  k={k}: |corr|={sym['abs_delta_corrected']:.4f}, "
                      f"|corrupt|={sym['abs_delta_corrupted']:.4f}, p={sym['p_value']:.4f}")


if __name__ == "__main__":
    main()
