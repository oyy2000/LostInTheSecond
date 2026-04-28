#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

from tqdm.auto import tqdm
from src.prm.scoring import (
    StepScorer,
    plot_avg_score_per_step,
    plot_baseline,
    plot_per_step_avg_correctness,
    print_avg_score_per_step,
    split_steps,
)


@dataclass
class Sample:
    doc_id: int
    question: str
    answer: str
    target_solution: str
    generation: str
    exact_match: float
    n_tokens: int
    step1: str
    step2: str
    generation_tail: str
    step2_score: float
    step_scores: List[float]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_latest_baseline_samples(run_root: Path) -> Path:
    cands = sorted(run_root.rglob("samples_*.jsonl"))
    if not cands:
        raise FileNotFoundError(f"No samples_*.jsonl found under {run_root}")
    return cands[-1]


def write_scores_json(samples: List[Sample], out_json: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples": [
            {
                "doc_id": s.doc_id,
                "exact_match": float(s.exact_match),
                "n_tokens": int(s.n_tokens),
                "step2_score": float(s.step2_score),
                "step_scores": [float(x) for x in s.step_scores],
                "steps": split_steps(s.generation, mode="double_newline"), 
            }
            for s in samples
        ]
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_samples_from_scores_json(in_json: Path) -> List[Sample]:
    data = json.loads(in_json.read_text(encoding="utf-8"))
    raw = data.get("samples", [])
    out: List[Sample] = []
    for x in raw:
        out.append(
            Sample(
                doc_id=int(x.get("doc_id", -1)),
                question="",
                answer="",
                target_solution="",
                generation="\n\n".join(x.get("steps", []) or []),
                exact_match=float(x.get("exact_match", 0.0)),
                n_tokens=int(x.get("n_tokens", 0)),
                step1="",
                step2="",
                generation_tail="",
                step2_score=float(x.get("step2_score", -1e9)),
                step_scores=[float(v) for v in (x.get("step_scores", []) or [])],
            )
        )
    return out


def store(sample: Sample) -> Dict:
   

    return {
        "doc": {"question": sample.question, "id": sample.doc_id},
        "neg_response": sample.generation,
        "results": {"exact_match": 0.0},
    }


def main():
    parser = argparse.ArgumentParser(description="Phase1+2: score baseline and build DS1/DS2")
    parser.add_argument("--baseline-samples", default="", help="samples_*.jsonl; if empty, auto-find in --runs-root")
    parser.add_argument("--runs-root", default="./runs/baseline_qwen25_3b_math500")
    parser.add_argument("--score-model", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--score-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument(
        "--step-split-mode",
        default="double_newline",
        choices=["double_newline", "single_newline", "auto", "regex"],
        help="Step segmentation mode. README recommendation: double_newline",
    )
    parser.add_argument("--max-subset", type=int, default=100)
    parser.add_argument("--artifacts-dir", default="./artifacts")
    parser.add_argument("--out-ids", default="./artifacts/s_step2_wrong_ids.json")
    parser.add_argument("--out-prm-json", default="./artifacts/prm_scores_baseline.json")
    parser.add_argument("--out-neg", default="./artifacts/negative_samples_step2_wrong.json")
    parser.add_argument("--reuse-prm-json", action="store_true", help="Reuse existing --out-prm-json to plot/export without rerunning PRM")
    parser.add_argument("--only-plot", action="store_true", help="Only generate plots from cached PRM json")
    parser.add_argument("--plot-threshold", type=float, default=0.72)
    args = parser.parse_args()

    prm_json_path = Path(args.out_prm_json)
    use_cache = args.reuse_prm_json or args.only_plot

    if use_cache and prm_json_path.exists():
        print(f"[Load] reuse cached PRM scores: {prm_json_path}")
        samples = load_samples_from_scores_json(prm_json_path)
    else:
        baseline_path = Path(args.baseline_samples) if args.baseline_samples else find_latest_baseline_samples(Path(args.runs_root))
        print(f"[Load] baseline samples: {baseline_path}")

        scorer = StepScorer(args.score_model, args.score_dtype)

        samples: List[Sample] = []
        for rec in tqdm(read_jsonl(baseline_path), desc="PRM scoring", unit="sample"):
            doc = rec.get("doc", {}) or {}
            question = (doc.get("problem") or doc.get("question") or "").strip()
            answer = (doc.get("answer") or "").strip()
            target_solution = (doc.get("solution") or rec.get("target") or "").strip()

            gen = ""
            fr = rec.get("filtered_resps", [])
            if isinstance(fr, list) and fr:
                gen = (fr[0] or "").strip()
            if not gen:
                rs = rec.get("resps", [])
                if rs and isinstance(rs[0], list) and rs[0]:
                    gen = (rs[0][0] or "").strip()

            exact = float(rec.get("exact_match", 0.0))
            if not question or not gen or not target_solution:
                continue

            steps = split_steps(gen, mode=args.step_split_mode)
            step1 = steps[0] if len(steps) >= 1 else ""
            step2 = steps[1] if len(steps) >= 2 else ""
            generation_tail = "\n".join(steps[2:]).strip() if len(steps) > 2 else ""

            query_text = (
                (((rec.get("arguments") or {}).get("gen_args_0") or {}).get("arg_0"))
                or question
            )
            step_scores = scorer.score_steps(query_text, steps if steps else [gen])
            if len(step_scores) >= 2:
                step2_score = float(step_scores[1])
            elif step_scores:
                step2_score = float(step_scores[0])
            else:
                step2_score = -1e9

            n_tokens = len(scorer.tok(gen, add_special_tokens=False).input_ids)

            samples.append(
                Sample(
                    doc_id=int(rec.get("doc_id", -1)),
                    question=question,
                    answer=answer,
                    target_solution=target_solution,
                    generation=gen,
                    exact_match=exact,
                    n_tokens=n_tokens,
                    step1=step1,
                    step2=step2,
                    generation_tail=generation_tail,
                    step2_score=step2_score,
                    step_scores=step_scores,
                )
            )

    if not samples:
        raise ValueError("No valid baseline samples parsed.")

    artifacts_dir = Path(args.artifacts_dir)
    if not (use_cache and prm_json_path.exists()):
        write_scores_json(samples, prm_json_path)
    plot_baseline(samples, artifacts_dir / "prm_plots")
    print_avg_score_per_step(samples, "all_samples")
    plot_avg_score_per_step(samples, artifacts_dir / "prm_plots", "all_samples")
    plot_per_step_avg_correctness(
        samples,
        artifacts_dir / "prm_plots",
        "Per-step Avg Correctness | Reused PRM" if use_cache else "Per-step Avg Correctness",
        threshold=args.plot_threshold,
    )

    if args.only_plot:
        print("[Done] only-plot mode finished.")
        print(f"[Done] used PRM scores(json) -> {prm_json_path}")
        return

    wrong = [s for s in samples if s.exact_match < 1.0]
    if not wrong:
        raise ValueError("No wrong baseline samples, cannot build Step-2-wrong subset.")

    thr = 0.9
    subset_pool = [s for s in wrong if s.step2_score <= thr]
    subset = list(reversed(subset_pool))[: args.max_subset]

    print_avg_score_per_step(subset, "subset")
    plot_avg_score_per_step(subset, artifacts_dir / "prm_plots", "subset")

    out_ids = Path(args.out_ids)
    out_ids.parent.mkdir(parents=True, exist_ok=True)
    out_ids.write_text(
        json.dumps({"ids": [s.doc_id for s in subset]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    neg_out = {"samples": [store(s) for s in subset]}

    out_neg = Path(args.out_neg)
    out_neg.parent.mkdir(parents=True, exist_ok=True)
    out_neg.write_text(json.dumps(neg_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Done] parsed={len(samples)} wrong={len(wrong)} subset={len(subset)}")
    print(f"[Done] PRM scores(json) -> {Path(args.out_prm_json)}")
    print(f"[Done] subset ids -> {out_ids}")
    print(f"[Done] negative samples -> {out_neg}")


if __name__ == "__main__":
    main()
