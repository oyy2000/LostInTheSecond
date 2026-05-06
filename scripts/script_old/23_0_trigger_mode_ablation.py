#!/usr/bin/env python3
"""
Experiment 3: Lookback vs Lookahead vs Symmetric trigger ablation.

Compares three trigger modes at matched compute:
  A) Lookahead: entropy at step t is high -> repair step t
  B) Lookback:  logprob drop at step t -> repair step t-1
  C) Symmetric: trigger at step t -> repair both t-1 and t

For each mode, sweeps thresholds to produce Pareto points (tokens, accuracy).
Shares generation infrastructure with Experiment 1 (21_0_pareto_sweep.py).

Usage:
    python scripts/23_0_trigger_mode_ablation.py --gpus 0,1,2,3,4,5,6,7
    python scripts/23_0_trigger_mode_ablation.py --model-id Qwen/Qwen2.5-3B-Instruct --dataset gsm8k
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = [
    ("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2-3B"),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen2.5-3B"),
]
DATASETS = [("gsm8k", "GSM8K"), ("math500", "MATH500")]
FIG_DIR = ROOT / "figures" / "trigger_ablation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ENTROPY_PCTS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 95]
REPAIR_K = 3
MAX_REPAIRS = 3
BATCH_PER_GPU = 6000

TRIGGER_CONFIGS = [
    {"mode": "lookahead", "metric": "mean_entropy", "label": "Lookahead (entropy)",
     "color": "#55A868", "marker": "v", "ls": "-."},
    {"mode": "lookback", "metric": "logprob_drop", "label": "Lookback (logprob_drop)",
     "color": "#C44E52", "marker": "^", "ls": "-"},
    {"mode": "symmetric", "metric": "mean_entropy", "label": "Symmetric (entropy)",
     "color": "#4C72B0", "marker": "D", "ls": "--"},
]


def _ms(mid):
    return mid.split("/")[-1].lower().replace("-", "_")


def _load_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text("utf-8").splitlines() if l.strip()]


def _write_jsonl(p, recs):
    with p.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _vote(answers):
    if not answers:
        return ""
    return Counter(answers).most_common(1)[0][0]


def _batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _answer_kw(q):
    kw = {}
    if "test" in q:
        kw["test"] = q["test"]
        kw["entry_point"] = q.get("entry_point", "")
    return kw


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--_sid", type=int, default=-1)
    ap.add_argument("--_gpu", default="")
    ap.add_argument("--_tf", default="")
    ap.add_argument("--_out", default="")
    ap.add_argument("--_phase", default="")
    return ap.parse_args()


def run_shard_generation(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu
    from vllm import LLM, SamplingParams
    from src.prompt_templates import get_stop_tokens, extract_answer

    tasks = json.loads(Path(args._tf).read_text("utf-8"))
    stop = get_stop_tokens(args.model_id)
    llm = LLM(model=args.model_id, tensor_parallel_size=1,
              trust_remote_code=True, dtype="half",
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, stop=stop)
    outputs = llm.generate([t["prompt"] for t in tasks], sp)
    out_records = []
    for task, output in zip(tasks, outputs):
        text = output.outputs[0].text.strip()
        n_tok = len(output.outputs[0].token_ids)
        full = task["prefix_text"] + "\n\n" + text
        pred = extract_answer(args.dataset, full)
        out_records.append({
            "doc_id": task["doc_id"], "repair_step": task["repair_step"],
            "suffix_idx": task["suffix_idx"], "config_tag": task.get("config_tag", ""),
            "suffix_text": text, "pred_answer": pred, "suffix_tokens": n_tok,
        })
    Path(args._out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records),
        encoding="utf-8")


def _launch(script, args, tasks, sd, gpus, phase):
    if not tasks:
        return []
    ns = len(gpus)
    shards = [[] for _ in range(ns)]
    for i, t in enumerate(tasks):
        shards[i % ns].append(t)
    procs, outs = [], []
    for si, gid in enumerate(gpus):
        if not shards[si]:
            continue
        tf = sd / f"t_{phase}_{si}.json"
        tf.write_text(json.dumps(shards[si], ensure_ascii=False), encoding="utf-8")
        of = sd / f"o_{phase}_{si}.jsonl"
        outs.append(of)
        lf = sd / f"log_{phase}_{si}.txt"
        lh = open(lf, "w")
        cmd = [sys.executable, script,
               "--model-id", args.model_id, "--dataset", args.dataset,
               "--temperature", str(args.temperature),
               "--top-p", str(args.top_p),
               "--max-tokens", str(args.max_tokens),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--max-model-len", str(args.max_model_len),
               "--batch-size", str(args.batch_size),
               "--_sid", str(si), "--_gpu", gid,
               "--_tf", str(tf), "--_out", str(of), "--_phase", phase]
        p = subprocess.Popen(cmd, stdout=lh, stderr=subprocess.STDOUT)
        procs.append((si, gid, p, lh, lf))
    for si, gid, p, lh, lf in procs:
        p.wait()
        lh.close()
        rc = p.returncode
        if rc != 0:
            print(f"  WARNING: shard {si} failed (exit={rc})")
        lf.unlink(missing_ok=True)
    recs = []
    for sf in outs:
        if sf.exists():
            recs.extend(_load_jsonl(sf))
            sf.unlink(missing_ok=True)
    for si in range(ns):
        (sd / f"t_{phase}_{si}.json").unlink(missing_ok=True)
    return recs


def run_one_combo(model_id, dataset, args):
    from src.prompt_templates import build_prompt, split_steps, check_answer
    from src.repair_engine import should_trigger, METRIC_DIRECTION
    from src.sweep_datasets import load_dataset_by_name
    from src.pareto_engine import extract_pareto_front

    ms = _ms(model_id)
    pareto_dir = ROOT / "results" / f"{dataset}_{ms}_pareto"
    out_dir = ROOT / "results" / f"{dataset}_{ms}_trigger_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = out_dir / "_shards"
    sd.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    script = str(Path(__file__).resolve())
    args.model_id = model_id
    args.dataset = dataset

    questions = load_dataset_by_name(dataset, 0, 42)
    nq = len(questions)

    # Load drafts + step metrics from pareto sweep
    drafts = _load_jsonl(pareto_dir / "drafts.jsonl")
    sm_records = _load_jsonl(pareto_dir / "step_metrics.jsonl")
    if not drafts or not sm_records:
        print(f"  No pareto data for {model_id} x {dataset}, skipping")
        return None

    drafts_by_q = {}
    for d in drafts:
        drafts_by_q.setdefault(d["doc_id"], []).append(d)
    sm_map = {r["doc_id"]: r["step_metrics"] for r in sm_records}

    suffix_file = out_dir / "ablation_suffixes.jsonl"
    if suffix_file.exists():
        all_suffixes = _load_jsonl(suffix_file)
        print(f"  Suffixes cached: {len(all_suffixes)}")
    else:
        all_tasks = []
        for tcfg in TRIGGER_CONFIGS:
            mode = tcfg["mode"]
            metric = tcfg["metric"]
            higher = METRIC_DIRECTION.get(metric, True)

            all_vals = []
            for sm in sm_map.values():
                for m in sm:
                    v = m.get(metric)
                    if v is not None:
                        all_vals.append(v)
            if not all_vals:
                continue
            arr = np.array(all_vals)

            for pct in ENTROPY_PCTS:
                if higher:
                    thr = float(np.percentile(arr, 100 - pct))
                else:
                    thr = float(np.percentile(arr, pct))

                tag = f"{mode}_{metric}_pct{pct}"
                for q in questions:
                    did = q["doc_id"]
                    sm = sm_map.get(did, [])
                    ds = drafts_by_q.get(did, [])
                    if not sm or not ds or len(sm) < 2:
                        continue
                    d0 = ds[0]
                    steps = d0.get("draft_steps") or split_steps(
                        d0.get("draft_text", ""))
                    if not steps:
                        continue

                    repairs = set()
                    for t in range(len(sm)):
                        if len(repairs) >= MAX_REPAIRS:
                            break
                        val = sm[t].get(metric, 0.0)
                        if not should_trigger(val, thr, metric):
                            continue
                        if mode == "lookback":
                            if t - 1 >= 0:
                                repairs.add(t - 1)
                        elif mode == "lookahead":
                            repairs.add(t)
                        elif mode == "symmetric":
                            if t - 1 >= 0:
                                repairs.add(t - 1)
                            repairs.add(t)
                    repairs = sorted(repairs)[:MAX_REPAIRS]
                    if not repairs:
                        continue

                    prompt_base = build_prompt(model_id, dataset, q["question"])
                    for rs in repairs:
                        b = max(1, min(rs, len(steps) - 1))
                        prefix = "\n\n".join(steps[:b])
                        prompt = prompt_base + prefix + "\n\n"
                        for si in range(REPAIR_K):
                            all_tasks.append({
                                "task_type": "repair_suffix",
                                "doc_id": did, "repair_step": rs,
                                "suffix_idx": si, "gold_answer": q["gold_answer"],
                                "prefix_text": prefix, "prompt": prompt,
                                "mode": "sampled", "config_tag": tag,
                            })

        print(f"  Generating {len(all_tasks)} ablation suffixes...")
        all_suffixes = []
        for batch in _batched(all_tasks, BATCH_PER_GPU * len(gpus)):
            new = _launch(script, args, batch, sd, gpus, "ablation")
            all_suffixes.extend(new)
        _write_jsonl(suffix_file, all_suffixes)

    # Evaluate
    sfx_by_tag = {}
    for s in all_suffixes:
        sfx_by_tag.setdefault(s.get("config_tag", ""), []).append(s)

    results_by_mode = {}
    for tag, sfx_list in sorted(sfx_by_tag.items()):
        sfx_by_q = {}
        for s in sfx_list:
            sfx_by_q.setdefault(s["doc_id"], []).append(s)
        nc, total_tok = 0, 0.0
        for q in questions:
            did = q["doc_id"]
            kw = _answer_kw(q)
            ds = drafts_by_q.get(did, [])
            if not ds:
                continue
            d0 = ds[0]
            answers = [d0.get("draft_answer", "")]
            total_tok += d0.get("draft_tokens", 0)
            for s in sfx_by_q.get(did, []):
                answers.append(s.get("pred_answer", ""))
                total_tok += s.get("suffix_tokens", 0)
            if check_answer(dataset, _vote(answers), q["gold_answer"], **kw):
                nc += 1
        acc = nc / nq if nq else 0
        tpq = total_tok / nq if nq else 0
        mode = tag.split("_")[0]
        results_by_mode.setdefault(mode, []).append({
            "tag": tag, "accuracy": acc, "mean_tokens": tpq,
        })

    # Greedy baseline
    nc, gt = 0, 0.0
    for q in questions:
        ds = drafts_by_q.get(q["doc_id"], [])
        if ds:
            d0 = ds[0]
            if check_answer(dataset, d0.get("draft_answer", ""),
                            q["gold_answer"], **_answer_kw(q)):
                nc += 1
            gt += d0.get("draft_tokens", 0)
    greedy_acc = nc / nq if nq else 0
    greedy_tok = gt / nq if nq else 0

    summary = {
        "model": model_id, "dataset": dataset, "n_questions": nq,
        "greedy": {"accuracy": greedy_acc, "mean_tokens": greedy_tok},
        "results_by_mode": {
            mode: sorted(pts, key=lambda x: x["mean_tokens"])
            for mode, pts in results_by_mode.items()
        },
    }
    (out_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        sd.rmdir()
    except OSError:
        pass
    return summary


def plot_ablation(summaries):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Trigger Mode Ablation: Lookback vs Lookahead vs Symmetric",
                 fontsize=14, fontweight="bold", y=1.01)
    from src.pareto_engine import extract_pareto_front

    idx = 0
    for ri, (mid, ml) in enumerate(MODELS):
        for ci, (ds, dl) in enumerate(DATASETS):
            ax = axes[ri, ci]
            key = f"{_ms(mid)}_{ds}"
            s = summaries.get(key)
            if not s:
                ax.set_title(f"{ml} / {dl} (no data)")
                continue
            ax.scatter([s["greedy"]["mean_tokens"]], [s["greedy"]["accuracy"]],
                       color="black", marker="*", s=200, zorder=10, label="Greedy")
            for tcfg in TRIGGER_CONFIGS:
                mode = tcfg["mode"]
                pts = s.get("results_by_mode", {}).get(mode, [])
                if not pts:
                    continue
                raw = [(p["mean_tokens"], p["accuracy"]) for p in pts]
                front = extract_pareto_front(raw)
                if front:
                    xs, ys = zip(*front)
                    ax.plot(xs, ys, color=tcfg["color"], marker=tcfg["marker"],
                            ls=tcfg["ls"], lw=2, markersize=5, alpha=0.85,
                            label=tcfg["label"])
            ax.set_title(f"{ml} / {dl}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Total Tokens (per question)", fontsize=9)
            ax.set_ylabel("Accuracy", fontsize=9)
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig_trigger_ablation.{ext}",
                    dpi=200, bbox_inches="tight")
    print(f"Saved fig_trigger_ablation to {FIG_DIR}")
    plt.close(fig)


def main():
    args = parse_args()
    if args._sid >= 0:
        run_shard_generation(args)
        return

    summaries = {}
    models = MODELS
    datasets = DATASETS
    if args.model_id:
        models = [(args.model_id, args.model_id.split("/")[-1])]
    if args.dataset:
        datasets = [(args.dataset, args.dataset.upper())]

    for mid, ml in models:
        for ds, dl in datasets:
            print(f"\n{'='*60}\n  {ml} x {dl}\n{'='*60}")
            s = run_one_combo(mid, ds, args)
            if s:
                summaries[f"{_ms(mid)}_{ds}"] = s

    if summaries:
        plot_ablation(summaries)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
