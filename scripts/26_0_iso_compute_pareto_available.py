#!/usr/bin/env python3
"""
Build an iso-compute Pareto figure from existing completed result files.

This script does not invent missing repair runs. It plots the method families
that are already materialized in the repository and explicitly records missing
families such as Random-repair and Entropy-triggered lookback repair when their
actual suffix-generation summaries are unavailable.

Outputs:
  figures/pareto/fig_iso_compute_pareto_available.{png,pdf}
  results/iso_compute_pareto_available_summary.json
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pareto_engine import extract_pareto_front
from src.prompt_templates import check_answer

FIG_DIR = ROOT / "figures" / "pareto"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SC_NS = [1, 2, 4, 8, 16, 32]


PANELS = [
    ("meta-llama/Llama-3.2-3B-Instruct", "llama_3.2_3b_instruct", "gsm8k", "Llama-3.2-3B / GSM8K"),
    ("meta-llama/Llama-3.2-3B-Instruct", "llama_3.2_3b_instruct", "math500", "Llama-3.2-3B / MATH500"),
    ("Qwen/Qwen2.5-3B-Instruct", "qwen2.5_3b_instruct", "gsm8k", "Qwen2.5-3B / GSM8K"),
    ("Qwen/Qwen2.5-3B-Instruct", "qwen2.5_3b_instruct", "math500", "Qwen2.5-3B / MATH500"),
]


STYLE = {
    "Greedy": {"color": "#222222", "marker": "*", "ls": "None", "ms": 13, "label": "Greedy"},
    "Self-Consistency": {"color": "#4C72B0", "marker": "o", "ls": "-", "ms": 5, "label": "Self-Consistency"},
    "LateRollback": {"color": "#8172B2", "marker": "^", "ls": "-", "ms": 5, "label": "Late rollback (available)"},
    "RandomRepair": {"color": "#DD8452", "marker": "s", "ls": "--", "ms": 5, "label": "Random repair"},
    "EntropyLookback": {"color": "#C44E52", "marker": "D", "ls": "-", "ms": 5, "label": "Entropy lookback"},
}


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def add_point(points: List[dict], method: str, x: float, y: float, **meta) -> None:
    if x is None or y is None or x <= 0:
        return
    rec = {"method": method, "mean_tokens": float(x), "accuracy": float(y)}
    rec.update(meta)
    points.append(rec)


def evaluate_sc_from_records(dataset: str, records: List[dict], ns: List[int]) -> List[dict]:
    by_q: Dict[Any, List[dict]] = defaultdict(list)
    gold_by_q: Dict[Any, Any] = {}
    for rec in records:
        by_q[rec["doc_id"]].append(rec)
        gold_by_q[rec["doc_id"]] = rec.get("gold_answer", "")
    out = []
    nq = len(by_q)
    for n in ns:
        ok = 0
        total_tok = 0.0
        covered = 0
        for did, samples in by_q.items():
            chunk = samples[:n]
            if len(chunk) < n:
                continue
            covered += 1
            answers = [
                s.get("pred_answer")
                or s.get("sc_answer")
                or s.get("draft_answer")
                or ""
                for s in chunk
            ]
            vote = Counter(answers).most_common(1)[0][0] if answers else ""
            total_tok += sum(
                s.get("n_tokens")
                or s.get("tokens")
                or s.get("sc_tokens")
                or s.get("draft_tokens")
                or 0
                for s in chunk
            )
            if check_answer(dataset, vote, gold_by_q[did]):
                ok += 1
        if covered:
            out.append({
                "N": n,
                "accuracy": ok / nq if nq else 0.0,
                "mean_tokens": total_tok / nq if nq else 0.0,
                "covered_questions": covered,
                "n_questions": nq,
            })
    return out


def greedy_from_drafts(dataset: str, path: Path, doc_limit: int = 0) -> dict:
    records = load_jsonl(path)
    by_q = {}
    for rec in records:
        if rec.get("task_type") not in ("draft", None):
            continue
        idx = rec.get("draft_idx", rec.get("sample_idx", 0))
        mode = rec.get("mode", "greedy")
        if idx != 0 and mode != "greedy":
            continue
        did = rec.get("doc_id")
        if did not in by_q:
            by_q[did] = rec
    if doc_limit:
        by_q = dict(list(by_q.items())[:doc_limit])
    if not by_q:
        return {}
    ok = 0
    tok = 0.0
    for rec in by_q.values():
        pred = rec.get("draft_answer") or rec.get("pred_answer") or ""
        gold = rec.get("gold_answer", "")
        ok += int(check_answer(dataset, pred, gold))
        tok += rec.get("draft_tokens") or rec.get("n_tokens") or 0
    return {"accuracy": ok / len(by_q), "mean_tokens": tok / len(by_q), "n_questions": len(by_q)}


def qwen_gsm8k_points() -> Dict[str, Any]:
    points: List[dict] = []
    missing = ["RandomRepair", "EntropyLookback"]
    grid_path = ROOT / "results/gsm8k_3b_multi_sample/grid_search/grid_summary.json"
    grid = json.loads(grid_path.read_text("utf-8"))
    add_point(
        points, "Greedy",
        grid["greedy_tokens_per_question"],
        grid["greedy_accuracy"],
        source=str(grid_path),
    )
    sc_records = load_jsonl(ROOT / "results/gsm8k_3b_multi_sample/grid_search/full_sc_samples.jsonl")
    for rec in evaluate_sc_from_records("gsm8k", sc_records, SC_NS):
        add_point(points, "Self-Consistency", rec["mean_tokens"], rec["accuracy"], N=rec["N"])
    for name, rec in grid.get("grid", {}).items():
        if rec.get("method") == "late_rollback":
            add_point(
                points, "LateRollback",
                rec["tokens_per_question"], rec["accuracy"],
                K=rec.get("K"), alpha=rec.get("alpha"), config=name,
            )
    return {"points": points, "missing_methods": missing}


def qwen_math500_points() -> Dict[str, Any]:
    points: List[dict] = []
    missing = ["RandomRepair", "EntropyLookback"]
    summary_path = ROOT / "results/math500_full_sweep/sweep_summary.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    add_point(points, "Greedy", summary["greedy_tpq"], summary["greedy_accuracy"], source=str(summary_path))
    sc_records = load_jsonl(ROOT / "results/math500_full_sweep/full_sc.jsonl")
    for rec in evaluate_sc_from_records("math500", sc_records, SC_NS):
        add_point(points, "Self-Consistency", rec["mean_tokens"], rec["accuracy"], N=rec["N"])
    for rec in summary.get("results", []):
        if rec.get("method") == "LateRollback":
            add_point(
                points, "LateRollback",
                rec["tpq"], rec["accuracy"],
                K=rec.get("K"), alpha=rec.get("alpha"),
                n_drafts=rec.get("n_drafts"),
            )
    return {"points": points, "missing_methods": missing}


def collect_panel(model_short: str, dataset: str) -> Dict[str, Any]:
    if model_short == "qwen2.5_3b_instruct" and dataset == "gsm8k":
        data = qwen_gsm8k_points()
    elif model_short == "qwen2.5_3b_instruct" and dataset == "math500":
        data = qwen_math500_points()
    else:
        data = {"points": [], "missing_methods": ["Greedy", "Self-Consistency", "RandomRepair", "EntropyLookback"]}
    data["model_short"] = model_short
    data["dataset"] = dataset
    return data


def plot_panel(ax, panel: Dict[str, Any], title: str) -> None:
    points = panel["points"]
    if not points:
        ax.set_title(f"{title} (no local results)", fontsize=11, fontweight="bold")
        ax.text(0.5, 0.5, "No local result files", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=11)
        ax.set_xlabel("Total generation tokens / question")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.25)
        return
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for p in points:
        grouped[p["method"]].append(p)

    for method in ["Self-Consistency", "RandomRepair", "LateRollback", "EntropyLookback", "Greedy"]:
        recs = grouped.get(method, [])
        if not recs:
            continue
        style = STYLE[method]
        pts = sorted(
            ((r["mean_tokens"], r["accuracy"], r) for r in recs),
            key=lambda item: (item[0], item[1], str(item[2].get("N", "")),
                              str(item[2].get("K", "")),
                              str(item[2].get("alpha", ""))),
        )
        if method == "Greedy":
            x, y, _ = pts[0]
            ax.scatter([x], [y], color=style["color"], marker=style["marker"],
                       s=style["ms"] ** 2, label=style["label"], zorder=8)
        else:
            front_xy = extract_pareto_front([(x, y) for x, y, _ in pts])
            if front_xy:
                xs, ys = zip(*front_xy)
                ax.plot(xs, ys, color=style["color"], marker=style["marker"],
                        linestyle=style["ls"], linewidth=2.0, markersize=style["ms"],
                        label=style["label"])
            non_front = [(x, y) for x, y, _ in pts if (x, y) not in set(front_xy)]
            if non_front:
                xs, ys = zip(*non_front)
                ax.scatter(xs, ys, color=style["color"], marker=style["marker"],
                           s=18, alpha=0.25)

    missing = [m for m in panel.get("missing_methods", []) if m in ("RandomRepair", "EntropyLookback")]
    if missing:
        ax.text(0.02, 0.02, "Missing actual runs: " + ", ".join(missing),
                transform=ax.transAxes, fontsize=7, color="#7a3b00",
                ha="left", va="bottom")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Total generation tokens / question")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")


def main() -> None:
    summary = {"note": "Available local results only; missing methods are not imputed.", "panels": {}}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    for ax, (_, model_short, dataset, title) in zip(axes.ravel(), PANELS):
        panel = collect_panel(model_short, dataset)
        summary["panels"][f"{model_short}_{dataset}"] = panel
        plot_panel(ax, panel, title)
    fig.suptitle("Iso-Compute Pareto Curves (Available Local Results)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_iso_compute_pareto_available.{ext}", dpi=220, bbox_inches="tight")
    (ROOT / "results/iso_compute_pareto_available_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved -> {FIG_DIR / 'fig_iso_compute_pareto_available.png'}")
    print(f"Saved -> {ROOT / 'results/iso_compute_pareto_available_summary.json'}")
    for key, panel in summary["panels"].items():
        print(key, "points", len(panel["points"]), "missing", panel["missing_methods"])


if __name__ == "__main__":
    main()
