#!/usr/bin/env python3
"""
Multi-GPU PRM scoring + plotting for LostInTheSecond steering experiments.

Usage:
    python 06_multi_card_PRM.py                 # run PRM scoring + merge + plot
    python 06_multi_card_PRM.py --only-plot      # skip scoring, just plot from merged JSON

Modelled after fact-enhancement/06_multi_card_PRM_sml.py + 00_PRM_boards.py
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm.auto import tqdm
from prm_shared import (
    PRMSampleLite,
    StepScorer,
    plot_avg_score_per_step,
    plot_baseline,
    plot_per_step_avg_correctness,
    print_avg_score_per_step,
    split_steps,
)

# ============================================================
# ===== CONFIG ==============================================
# ============================================================

# # --- Experiment directory ---
# RESULTS_ROOT = (
#     # "./artifacts/vectors_16_ds2_wait_recompute_incorrect_only/"
#     "./artifacts/vectors_16_ds2_fix_step2_incorrect_only/"
#     "Qwen_Qwen2.5-3B-Instruct_applied/"
#     "hendrycks_math_500_step2fix_fix_step2" 
#     # "hendrycks_math_500_step2fix_wait_recompute"
# )

RESULTS_ROOT = ("/common/users/sl2148/Public/yang_ouyang/projects/LostInTheSecond/runs/09_token_range_exp_N256/prefix")

# --- PRM model ---
PRM_MODEL = "Qwen/Qwen2.5-Math-PRM-7B"
PRM_DTYPE = "float16"

# --- GPU ---
NUM_GPUS = 8
GPU_IDS = list(range(NUM_GPUS))


NUM_GPUS = 3
GPU_IDS = [5, 6, 7]
# --- Generation model (used for tokenisation stats) ---
GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# --- Step split strategy ---
STEP_SPLIT_STRATEGY = "auto"  # "auto", "double_newline", "single_newline"

# --- PRM scoring ---
THRESHOLD = 0.9
MAX_STEPS_FOR_EXTRA_PLOT = 10

# --- Output ---
PRM_OUT_DIR = os.path.join(RESULTS_ROOT, "prm")
os.makedirs(PRM_OUT_DIR, exist_ok=True)

# ============================================================
# ===== HELPERS =============================================
# ============================================================

def lam_to_str(lam: float) -> str:
    if abs(lam) < 1e-9:
        return "BASELINE"
    sign = "-" if lam < 0 else ""
    s = f"{abs(lam):.2f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    s = s.replace(".", "p")
    return f"lam{sign}{s}"


def lam_to_float(lam_str: str) -> float:
    if lam_str == "BASELINE":
        return 0.0
    s = lam_str
    if s.startswith("lam"):
        s = s[3:]
    s = s.replace("p", ".")
    if s.startswith("m"):
        s = "-" + s[1:]
    return float(s)


def parse_run_dir_name(dirname: str):
    """Parse e.g. 'Qwen2.5-3B-Instruct_L16_lam1p0' -> (model, layer, lam_str)"""
    parts = dirname.rsplit("_", 2)
    if len(parts) < 3:
        # Could be BASELINE: 'Qwen2.5-3B-Instruct_L16_BASELINE'
        parts = dirname.rsplit("_", 2)
    if len(parts) == 3:
        model_short = parts[0]  # e.g. Qwen2.5-3B-Instruct
        layer_str = parts[1]    # e.g. L16
        lam_str = parts[2]      # e.g. lam1p0 or BASELINE
        return model_short, layer_str, lam_str
    return None, None, None


def find_samples_jsonl(run_dir: str) -> Optional[str]:
    """Find samples_*.jsonl inside a run directory (may be nested under model subfolder)."""
    pattern = os.path.join(run_dir, "**", "samples_*.jsonl")
    files = sorted(glob.glob(pattern, recursive=True))
    return files[-1] if files else None


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in str(s))


# ============================================================
# ===== DISCOVER JOBS =======================================
# ============================================================

def discover_jobs(results_root: str):
    """Walk the results_root and find all experiment sub-dirs with samples JSONL."""
    jobs = []
    root = Path(results_root)

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        dirname = entry.name
        model_short, layer_str, lam_str = parse_run_dir_name(dirname)
        if model_short is None:
            continue

        jsonl_path = find_samples_jsonl(str(entry))
        if jsonl_path is None:
            print(f"⚠ No samples JSONL found in {entry}")
            continue

        jobs.append({
            "model_short": model_short,
            "layer": layer_str,
            "lam": lam_str,
            "jsonl": jsonl_path,
            "dir": str(entry),
        })

    return jobs


# ============================================================
# ===== PER-GPU WORKER (single process, loads PRM once) ======
# ============================================================

def worker_batch(gpu_id: int, job_list: list, out_dir: str):
    """
    Process a batch of jobs on a single GPU.
    Loads the PRM model once, scores all assigned jobs sequentially.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from transformers import AutoTokenizer

    print(f"[GPU {gpu_id}] Loading PRM model: {PRM_MODEL}")
    scorer = StepScorer(PRM_MODEL, PRM_DTYPE)

    # Also load gen tokenizer for token-len stats
    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL, trust_remote_code=True, use_fast=True)

    for job in tqdm(job_list, desc=f"GPU{gpu_id} jobs", unit="job"):
        model_short = job["model_short"]
        layer = job["layer"]
        lam = job["lam"]
        jsonl_path = job["jsonl"]

        print(f"[GPU {gpu_id}] {model_short} {layer} {lam}")

        data = [json.loads(line) for line in open(jsonl_path) if line.strip()]

        samples_out = []

        for idx, d in enumerate(tqdm(data, desc=f"GPU{gpu_id} {model_short}_{layer}_{lam}", unit="sample", leave=False)):
            if d.get("filter") == "strict-match":
                continue

            cot = ""
            fr = d.get("filtered_resps", [])
            if isinstance(fr, list) and fr:
                cot = (fr[0] or "").strip()
            if not cot:
                rs = d.get("resps", [])
                if rs and isinstance(rs[0], list) and rs[0]:
                    cot = (rs[0][0] or "").strip()
            if not cot:
                continue

            steps = split_steps(cot, mode=STEP_SPLIT_STRATEGY)
            if not steps:
                continue

            try:
                query_text = d.get("arguments", {}).get("gen_args_0", {}).get("arg_0", "")
                if not query_text:
                    query_text = d.get("doc", {}).get("problem", "")
                scores = scorer.score_steps(query_text, steps)
            except Exception as e:
                print(f"[GPU {gpu_id}] WARN PRM failed sample {idx}: {e}")
                continue

            if len(scores) != len(steps):
                # Length mismatch — skip
                continue

            step_token_lens = [
                len(gen_tokenizer.encode(s, add_special_tokens=False)) for s in steps
            ]
            samples_out.append(
                {
                    "doc_id": d.get("doc_id", -1),
                    "exact_match": int(d.get("exact_match", 0)),
                    "step_scores": scores,
                    "steps_text": steps,
                    "step_token_len": step_token_lens,
                    "generated_text": cot,
                }
            )

        result = {
            "model": model_short,
            "layer": layer,
            "lam": lam,
            "gen_model": GEN_MODEL,
            "prm_model": PRM_MODEL,
            "n_samples": len(samples_out),
            "samples": samples_out,
        }

        chunk_name = f"run_{_safe_name(model_short)}_{_safe_name(layer)}_{_safe_name(lam)}.json"
        chunk_path = os.path.join(out_dir, chunk_name)
        with open(chunk_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[GPU {gpu_id}] ✔ Saved {chunk_path}  ({len(samples_out)} samples)")

    print(f"[GPU {gpu_id}] Done with all {len(job_list)} jobs.")


# ============================================================
# ===== MERGE ===============================================
# ============================================================

def merge_chunks(out_dir: str) -> dict:
    """Merge all chunk JSON files into nested dict: model -> layer -> lam -> data"""
    merged = {}
    run_files = sorted(glob.glob(os.path.join(out_dir, "run_*.json")))
    for f in tqdm(run_files, desc="Merging PRM chunks", unit="file"):
        chunk = json.load(open(f))
        model = chunk["model"]
        layer = chunk["layer"]
        lam = chunk["lam"]

        if "samples" in chunk:
            samples = chunk["samples"]
        else:
            # backward compatibility for old columnar chunks
            ys = chunk.get("Y", [])
            doc_ids = chunk.get("doc_ids", [])
            step_scores = chunk.get("step_scores", [])
            steps_text = chunk.get("steps_text", [])
            step_token_len = chunk.get("step_token_len", [])
            generated_text = chunk.get("generated_text", [])
            samples = []
            for i, (y, ss, st, sl, gt) in enumerate(
                zip(ys, step_scores, steps_text, step_token_len, generated_text)
            ):
                samples.append(
                    {
                        "doc_id": doc_ids[i] if i < len(doc_ids) else -1,
                        "exact_match": int(y),
                        "step_scores": ss,
                        "steps_text": st,
                        "step_token_len": sl,
                        "generated_text": gt,
                    }
                )

        merged.setdefault(model, {})
        merged[model].setdefault(layer, {})
        merged[model][layer][lam] = {
            "samples": samples,
            "n_samples": len(samples),
            "gen_model": chunk["gen_model"],
            "prm_model": chunk["prm_model"],
        }

    merged_path = os.path.join(out_dir, "results_merged.json")
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"🎉 Merged → {merged_path}")
    return merged


# ============================================================
# ===== PLOTTING ============================================
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot_curve_pair(
    baseline_vals: List[float],
    target_vals: List[float],
    out_path: Path,
    title: str,
    ylabel: str,
    count: int,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not baseline_vals and not target_vals:
        return

    max_len = max(len(baseline_vals), len(target_vals))
    xs = list(range(1, max_len + 1))

    plt.figure(figsize=(8, 4.8))
    if baseline_vals:
        plt.plot(xs[: len(baseline_vals)], baseline_vals, marker="o", linewidth=2, label="Baseline")
    if target_vals:
        plt.plot(xs[: len(target_vals)], target_vals, marker="s", linewidth=2, label="Target")
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(f"{title} (n={count})")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_delta_curve(delta_vals: List[float], out_path: Path, title: str, count: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not delta_vals:
        return

    xs = list(range(1, len(delta_vals) + 1))
    plt.figure(figsize=(8, 4.8))
    plt.plot(xs, delta_vals, marker="o", linewidth=2)
    plt.axhline(y=0.0, linestyle="--", linewidth=1.5, color="gray")
    plt.xlabel("Aligned step")
    plt.ylabel("Target - Baseline PRM score")
    plt.title(f"{title} (n={count})")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_flip_step_correctness(summary: dict, out_dir: str):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for model, layer_map in summary.items():
        for layer, lam_map in layer_map.items():
            # layer-level summary across lambdas
            lam_labels = []
            c2w_counts = []
            w2c_counts = []
            for lam, block in sorted(lam_map.items(), key=lambda x: lam_to_float(x[0])):
                lam_labels.append(lam)
                c2w_counts.append(block["correct_to_wrong"]["n_samples"])
                w2c_counts.append(block["wrong_to_correct"]["n_samples"])

            if lam_labels:
                plt.figure(figsize=(max(8, len(lam_labels) * 0.9), 4.8))
                xs = np.arange(len(lam_labels))
                width = 0.38
                plt.bar(xs - width / 2, c2w_counts, width=width, label="Correct -> Wrong")
                plt.bar(xs + width / 2, w2c_counts, width=width, label="Wrong -> Correct")
                plt.xticks(xs, lam_labels, rotation=45, ha="right")
                plt.ylabel("Flip sample count")
                plt.title(f"Flip counts vs lambda | {model} {layer}")
                plt.grid(axis="y", alpha=0.25)
                plt.legend()
                plt.tight_layout()
                summary_path = out_root / model / layer / "flip_counts_summary.png"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(summary_path, dpi=180)
                plt.close()

            for lam, block in lam_map.items():
                lam_dir = out_root / model / layer / lam
                for flip_name, flip_title in [
                    ("correct_to_wrong", "Correct -> Wrong"),
                    ("wrong_to_correct", "Wrong -> Correct"),
                ]:
                    item = block[flip_name]
                    n = item["n_samples"]
                    if n <= 0:
                        continue

                    _plot_curve_pair(
                        baseline_vals=item["avg_baseline_step_correctness_hard"],
                        target_vals=item["avg_target_step_correctness_hard"],
                        out_path=lam_dir / f"{flip_name}_hard_correctness.png",
                        title=f"{flip_title} hard step correctness | {model} {layer} {lam}",
                        ylabel="Fraction of steps >= threshold",
                        count=n,
                    )
                    _plot_curve_pair(
                        baseline_vals=item["avg_baseline_step_score"],
                        target_vals=item["avg_target_step_score"],
                        out_path=lam_dir / f"{flip_name}_avg_step_score.png",
                        title=f"{flip_title} avg PRM step score | {model} {layer} {lam}",
                        ylabel="Average PRM score",
                        count=n,
                    )
                    _plot_delta_curve(
                        delta_vals=item["avg_delta_step_score_aligned"],
                        out_path=lam_dir / f"{flip_name}_delta_step_score.png",
                        title=f"{flip_title} aligned PRM score delta | {model} {layer} {lam}",
                        count=n,
                    )


def _build_plot_samples(entry: dict) -> List[PRMSampleLite]:
    samples: List[PRMSampleLite] = []

    if "samples" in entry:
        for it in entry.get("samples", []):
            step_scores = it.get("step_scores", [])
            if not step_scores:
                continue
            step_token_lens = it.get("step_token_len", [])
            step2_score = float(step_scores[1]) if len(step_scores) >= 2 else float(step_scores[0])
            n_tokens = int(sum(step_token_lens)) if step_token_lens else 0
            samples.append(
                PRMSampleLite(
                    exact_match=float(it.get("exact_match", 0.0)),
                    n_tokens=n_tokens,
                    step2_score=step2_score,
                    step_scores=[float(v) for v in step_scores],
                )
            )
        return samples

    # backward compatibility for old merged format
    ys = entry.get("Y", [])
    step_scores_list = entry.get("step_scores", [])
    step_token_len_list = entry.get("step_token_len", [])
    for y, step_scores, step_token_lens in zip(ys, step_scores_list, step_token_len_list):
        if not step_scores:
            continue
        step2_score = float(step_scores[1]) if len(step_scores) >= 2 else float(step_scores[0])
        n_tokens = int(sum(step_token_lens)) if step_token_lens else 0
        samples.append(
            PRMSampleLite(
                exact_match=float(y),
                n_tokens=n_tokens,
                step2_score=step2_score,
                step_scores=[float(v) for v in step_scores],
            )
        )
    return samples


def _entry_to_samples(entry: dict) -> List[dict]:
    """Normalize merged entry (new/old format) to list-of-sample dict."""
    if "samples" in entry:
        out = []
        for it in entry.get("samples", []):
            step_scores = it.get("step_scores", [])
            out.append(
                {
                    "doc_id": int(it.get("doc_id", -1)),
                    "exact_match": int(it.get("exact_match", 0)),
                    "step_scores": [float(v) for v in step_scores],
                }
            )
        return out

    ys = entry.get("Y", [])
    doc_ids = entry.get("doc_ids", [])
    step_scores = entry.get("step_scores", [])
    out = []
    for i, (y, ss) in enumerate(zip(ys, step_scores)):
        out.append(
            {
                "doc_id": int(doc_ids[i]) if i < len(doc_ids) else i,
                "exact_match": int(y),
                "step_scores": [float(v) for v in ss],
            }
        )
    return out


def _avg_curve_from_lists(step_scores_list: List[List[float]]) -> List[float]:
    if not step_scores_list:
        return []
    max_steps = max((len(x) for x in step_scores_list), default=0)
    out = []
    for i in range(max_steps):
        vals = [x[i] for x in step_scores_list if i < len(x)]
        out.append(float(sum(vals) / len(vals)) if vals else 0.0)
    return out


def _hard_curve_from_lists(step_scores_list: List[List[float]], thr: float) -> List[float]:
    if not step_scores_list:
        return []
    max_steps = max((len(x) for x in step_scores_list), default=0)
    out = []
    for i in range(max_steps):
        vals = [1.0 if x[i] >= thr else 0.0 for x in step_scores_list if i < len(x)]
        out.append(float(sum(vals) / len(vals)) if vals else 0.0)
    return out


def _diff_curve_from_pair_lists(tgt_scores_list: List[List[float]], base_scores_list: List[List[float]]) -> List[float]:
    n = min(len(tgt_scores_list), len(base_scores_list))
    if n <= 0:
        return []
    deltas = []
    for i in range(n):
        t = tgt_scores_list[i]
        b = base_scores_list[i]
        k = min(len(t), len(b))
        deltas.append([(t[j] - b[j]) for j in range(k)])
    return _avg_curve_from_lists(deltas)


def compare_step_correctness_vs_baseline(model_results: dict, out_dir: str, threshold: float):
    """
    Compare each lambda run against BASELINE and report only changed samples:
      - correct_to_wrong: baseline exact_match=1 -> target exact_match=0
      - wrong_to_correct: baseline exact_match=0 -> target exact_match=1
    """
    os.makedirs(out_dir, exist_ok=True)
    summary: Dict[str, dict] = {}

    for model, layer_map in model_results.items():
        summary.setdefault(model, {})
        for layer, lam_map in layer_map.items():
            baseline_key = None
            baseline_entry = None

            if "BASELINE" in lam_map:
                baseline_key = "BASELINE"
                baseline_entry = lam_map.get("BASELINE")
            elif "lam0p0" in lam_map:
                baseline_key = "lam0p0"
                baseline_entry = lam_map.get("lam0p0")
            else:
                cand = []
                for k in lam_map.keys():
                    try:
                        cand.append((abs(lam_to_float(k)), k))
                    except Exception:
                        continue
                if cand:
                    _, baseline_key = min(cand, key=lambda x: x[0])
                    baseline_entry = lam_map.get(baseline_key)

            if baseline_entry is None:
                continue
            baseline_samples = _entry_to_samples(baseline_entry)
            baseline_by_doc = {s["doc_id"]: s for s in baseline_samples}
            if not baseline_by_doc:
                continue

            print(f"[BaselineSelect] {model} | {layer} -> {baseline_key}")

            summary[model].setdefault(layer, {})

            for lam, target_entry in lam_map.items():
                if lam == baseline_key:
                    continue
                target_samples = _entry_to_samples(target_entry)
                flip_groups = {
                    "correct_to_wrong": [],
                    "wrong_to_correct": [],
                }

                for t in target_samples:
                    did = t["doc_id"]
                    b = baseline_by_doc.get(did)
                    if b is None:
                        continue
                    b_em = int(b["exact_match"])
                    t_em = int(t["exact_match"])
                    if b_em == t_em:
                        continue
                    if b_em == 1 and t_em == 0:
                        flip = "correct_to_wrong"
                    elif b_em == 0 and t_em == 1:
                        flip = "wrong_to_correct"
                    else:
                        continue

                    b_scores = b.get("step_scores", [])
                    t_scores = t.get("step_scores", [])
                    k = min(len(b_scores), len(t_scores))
                    flip_groups[flip].append(
                        {
                            "doc_id": did,
                            "baseline_exact_match": b_em,
                            "target_exact_match": t_em,
                            "baseline_step_scores": b_scores,
                            "target_step_scores": t_scores,
                            "delta_step_scores_aligned": [t_scores[i] - b_scores[i] for i in range(k)],
                        }
                    )

                lam_block = {}
                for flip_name, items in flip_groups.items():
                    b_lists = [it["baseline_step_scores"] for it in items]
                    t_lists = [it["target_step_scores"] for it in items]
                    lam_block[flip_name] = {
                        "n_samples": len(items),
                        "avg_baseline_step_score": _avg_curve_from_lists(b_lists),
                        "avg_target_step_score": _avg_curve_from_lists(t_lists),
                        "avg_delta_step_score_aligned": _diff_curve_from_pair_lists(t_lists, b_lists),
                        "avg_baseline_step_correctness_hard": _hard_curve_from_lists(b_lists, threshold),
                        "avg_target_step_correctness_hard": _hard_curve_from_lists(t_lists, threshold),
                        "samples": items,
                    }
                summary[model][layer][lam] = lam_block

    out_json = os.path.join(out_dir, "step_correctness_vs_baseline_flips.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_flip_step_correctness(summary, os.path.join(out_dir, "flip_plots"))
    print(f"🧪 Baseline flip-step analysis saved -> {out_json}")

    # concise console summary
    for model, layer_map in summary.items():
        for layer, lam_map in layer_map.items():
            for lam, block in lam_map.items():
                c2w = block["correct_to_wrong"]["n_samples"]
                w2c = block["wrong_to_correct"]["n_samples"]
                if c2w or w2c:
                    print(f"[FlipSummary] {model} | {layer} | {lam} : C->W={c2w}, W->C={w2c}")


def _extract_accuracy_and_count(entry: dict) -> tuple:
    """Return (accuracy, n) for one merged entry, supporting new/old formats."""
    if "samples" in entry:
        vals = [float(it.get("exact_match", 0.0)) for it in entry.get("samples", [])]
    else:
        vals = [float(v) for v in entry.get("Y", [])]
    if not vals:
        return 0.0, 0
    return float(sum(vals) / len(vals)), len(vals)


def _clip_samples_to_max_steps(samples: List[PRMSampleLite], max_steps: int) -> List[PRMSampleLite]:
    clipped: List[PRMSampleLite] = []
    for s in samples:
        clipped_scores = s.step_scores[:max_steps]
        if not clipped_scores:
            continue
        clipped.append(
            PRMSampleLite(
                exact_match=s.exact_match,
                n_tokens=s.n_tokens,
                step2_score=clipped_scores[1] if len(clipped_scores) >= 2 else clipped_scores[0],
                step_scores=clipped_scores,
            )
        )
    return clipped


def plot_accuracy_vs_lambda(model_results: dict, save_root: str):
    """Create one summary figure: accuracy vs lambda, one line per model-layer."""
    os.makedirs(save_root, exist_ok=True)

    series = []
    for model, layer_map in model_results.items():
        for layer, lam_map in layer_map.items():
            pts = []
            for lam, entry in lam_map.items():
                try:
                    lam_val = lam_to_float(lam)
                except Exception:
                    continue
                acc, n = _extract_accuracy_and_count(entry)
                if n <= 0:
                    continue
                pts.append((lam_val, acc, n, lam))
            if pts:
                pts.sort(key=lambda x: x[0])
                series.append((f"{model}|{layer}", pts))

    if not series:
        print("⚠ No valid points for accuracy-vs-lambda summary plot.")
        return

    plt.figure(figsize=(10, 6))
    for label, pts in series:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=label)

    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Lambda (summary)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_path = Path(save_root) / "accuracy_vs_lambda_summary.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"[Plot] saved summary -> {out_path}")


def plot_all_metrics(model_results: dict, save_root: str, thr: float):
    """Generate per-run plots by reusing 00_PRM_boards plotting functions."""
    os.makedirs(save_root, exist_ok=True)

    plot_accuracy_vs_lambda(model_results, save_root)

    tasks = []
    for model, layer_map in model_results.items():
        for layer, lam_map in layer_map.items():
            for lam, entry in lam_map.items():
                tasks.append((model, layer, lam, entry))

    for model, layer, lam, entry in tqdm(tasks, desc="Plotting groups", unit="group"):
        samples = _build_plot_samples(entry)
        if not samples:
            print(f"⚠ Skip empty plot group: {model}/{layer}/{lam}")
            continue

        group_dir = Path(save_root) / model / layer / lam
        plot_baseline(samples, group_dir)
        plot_avg_score_per_step(samples, group_dir, f"{model}_{layer}_{lam}")
        samples_10_steps = _clip_samples_to_max_steps(samples, MAX_STEPS_FOR_EXTRA_PLOT)
        if samples_10_steps:
            plot_avg_score_per_step(
                samples_10_steps,
                group_dir,
                f"{model}_{layer}_{lam}_first{MAX_STEPS_FOR_EXTRA_PLOT}steps",
            )
        plot_per_step_avg_correctness(
            samples,
            group_dir,
            f"Per-step Avg Correctness | {model} {layer} {lam}",
            threshold=thr,
        )
        if samples_10_steps:
            plot_per_step_avg_correctness(
                samples_10_steps,
                group_dir / f"first{MAX_STEPS_FOR_EXTRA_PLOT}steps",
                f"Per-step Avg Correctness (First {MAX_STEPS_FOR_EXTRA_PLOT} Steps) | {model} {layer} {lam}",
                threshold=thr,
            )
        print_avg_score_per_step(samples, f"{model}/{layer}/{lam}")



# ============================================================
# ===== MAIN ===============================================
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU PRM scoring + plotting")
    parser.add_argument("--results-root", default=RESULTS_ROOT,
                        help="Root directory containing experiment sub-dirs")
    parser.add_argument("--prm-out-dir", default=PRM_OUT_DIR,
                        help="Output directory for PRM chunks and merged results")
    parser.add_argument("--only-plot", action="store_true",
                        help="Skip PRM scoring; only generate plots from merged JSON")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS)
    parser.add_argument(
        "--compare-to-baseline",
        action="store_true",
        help="Analyze step correctness on samples whose exact_match changed vs BASELINE",
    )
    args = parser.parse_args()

    merged_path = os.path.join(args.prm_out_dir, "results_merged.json")

    if not args.only_plot:
        # --- Discover jobs ---
        jobs = discover_jobs(args.results_root)
        print(f"\n📋 Discovered {len(jobs)} experiment runs:")
        for j in jobs:
            print(f"   {j['model_short']} | {j['layer']} | {j['lam']}")

        if not jobs:
            print("❌ No jobs found. Check --results-root path.")
            return

        # --- Distribute jobs across GPUs ---
        gpu_buckets = [[] for _ in range(args.num_gpus)]
        for i, job in enumerate(jobs):
            gpu_buckets[i % args.num_gpus].append(job)

        print(f"\n🚀 Launching {args.num_gpus} GPU workers...")
        for gi, bucket in enumerate(gpu_buckets):
            print(f"   GPU {GPU_IDS[gi]}: {len(bucket)} jobs")

        # --- Launch workers ---
        processes = []
        for gi, bucket in enumerate(gpu_buckets):
            if not bucket:
                continue
            p = mp.Process(
                target=worker_batch,
                args=(GPU_IDS[gi], bucket, args.prm_out_dir)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print("\n✅ All GPU workers finished.")

        # --- Merge ---
        model_results = merge_chunks(args.prm_out_dir)
    else:
        print(f"📂 Loading merged results from {merged_path}")
        if not os.path.exists(merged_path):
            print(f"❌ {merged_path} not found. Run without --only-plot first.")
            return
        model_results = json.load(open(merged_path))

    # --- Plot ---
    print("\n📊 Generating plots...")
    plot_save_dir = os.path.join(args.prm_out_dir, "plots")
    plot_all_metrics(model_results, plot_save_dir, args.threshold)

    if args.compare_to_baseline:
        compare_step_correctness_vs_baseline(
            model_results=model_results,
            out_dir=args.prm_out_dir,
            threshold=args.threshold,
        )

    print(f"\n🎉 All done! Plots → {plot_save_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
