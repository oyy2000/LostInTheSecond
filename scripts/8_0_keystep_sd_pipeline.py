#!/usr/bin/env python3
"""
KeyStep-SD: Large-model one-step intervention, small-model continuation.

Pipeline phases:
  A  Small-model greedy draft (3B) with per-token logprobs
  B  Compute trigger scores; identify steps to upgrade
  C  Large-model single-step generation (7B) for triggered steps
  D  Acceptance gate on large-model steps
  E  Small-model continuation (3B) from modified prefixes
  F  Aggregate results, compute accuracy / token cost

Multi-GPU data-parallel via subprocess sharding (same pattern as 7_1).

Usage:
    python scripts/8_0_keystep_sd_pipeline.py --gpus 0,1,2,3,4,5,6,7
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
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.keystep_utils import (
    build_chat_prompt,
    build_large_step_prompt,
    build_prefix_prompt,
    extract_boxed_answer,
    load_config,
    load_jsonl,
    normalize_answer,
    split_steps,
    write_jsonl,
    compute_step_logprob_stats,
    step_token_boundaries,
)
from src.keystep_trigger import decide_triggers
from src.keystep_acceptance import (
    build_verify_prompt,
    decide_acceptance,
    followability_score,
    parse_verify_score,
    prefix_compatibility_score,
)

DEFAULT_RAW = str(
    PROJECT_ROOT / "results/gsm8k_3b_multi_sample/raw_cot_n8.jsonl"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="KeyStep-SD pipeline")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs/keystep_sd.yaml"))
    ap.add_argument("--raw-file", default=DEFAULT_RAW)
    ap.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "results/gsm8k_3b_multi_sample/keystep_sd"))
    ap.add_argument("--gpus", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-acceptance", action="store_true",
                    help="Skip acceptance gate (accept all large-model steps)")
    # internal shard args
    ap.add_argument("--_shard-id", type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument("--_shard-out", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_gpu-id", default="0", help=argparse.SUPPRESS)
    ap.add_argument("--_task-file", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_mode", default="", help=argparse.SUPPRESS)
    ap.add_argument("--_model-id", default="", help=argparse.SUPPRESS)
    return ap.parse_args()

def run_shard(args) -> None:
    """Worker process: load one model on one GPU, run prompts, write results."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args._gpu_id
    from vllm import LLM, SamplingParams

    tasks = json.loads(Path(args._task_file).read_text("utf-8"))
    mode = args._mode
    model_id = args._model_id
    cfg = load_config(args.config)
    gcfg = cfg.get("generation", {})
    print(f"[Shard {args._shard_id}] GPU {args._gpu_id}: "
          f"{len(tasks)} prompts, mode={mode}, model={model_id}")

    if mode == "draft":
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=gcfg.get("max_tokens", 1024),
            stop=gcfg.get("stop_tokens", ["<|im_end|>", "<|endoftext|>"]),
            logprobs=1,
        )
    elif mode == "large_step":
        sp = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            stop=gcfg.get("stop_tokens", ["<|im_end|>", "<|endoftext|>"]) + ["\n\n"],
        )
    elif mode == "verify":
        sp = SamplingParams(temperature=0.0, max_tokens=16)
    else:
        sp = SamplingParams(
            temperature=gcfg.get("temperature", 0.7),
            top_p=gcfg.get("top_p", 0.9),
            max_tokens=gcfg.get("max_tokens", 1024),
            stop=gcfg.get("stop_tokens", ["<|im_end|>", "<|endoftext|>"]),
            logprobs=1,
        )

    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
        max_model_len=cfg.get("max_model_len", 2048),
        dtype="half",
    )

    prompts = [t["prompt"] for t in tasks]
    outputs = llm.generate(prompts, sp)

    out_path = Path(args._shard_out)
    with out_path.open("w", encoding="utf-8") as fout:
        for task, output in zip(tasks, outputs):
            gen_text = output.outputs[0].text.strip()
            n_tokens = len(output.outputs[0].token_ids)
            rec = dict(task)
            rec.pop("prompt", None)
            rec["generated_text"] = gen_text
            rec["n_tokens"] = n_tokens

            if sp.logprobs and output.outputs[0].logprobs:
                lps = []
                offsets = []
                cum = 0
                for lp_dict in output.outputs[0].logprobs:
                    if lp_dict:
                        top = next(iter(lp_dict.values()))
                        lps.append(top.logprob)
                        offsets.append(cum)
                        decoded = top.decoded_token or ""
                        cum += len(decoded)
                rec["token_logprobs"] = lps
                rec["token_offsets"] = offsets

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[Shard {args._shard_id}] Done -> {out_path}")

def launch_shards(
    args, tasks: List[Dict], mode: str, model_id: str,
    shard_dir: Path, gpu_ids: List[str],
) -> List[Dict]:
    if not tasks:
        print(f"  [{mode}] Nothing to do (0 tasks)")
        return []
    n_shards = len(gpu_ids)
    shard_tasks = [[] for _ in range(n_shards)]
    for i, t in enumerate(tasks):
        shard_tasks[i % n_shards].append(t)

    script_path = str(Path(__file__).resolve())
    procs: List[Tuple[int, str, subprocess.Popen]] = []
    shard_out_files: List[Path] = []

    for si, gpu_id in enumerate(gpu_ids):
        if not shard_tasks[si]:
            continue
        task_file = shard_dir / f"tasks_{mode}_{si}.json"
        task_file.write_text(
            json.dumps(shard_tasks[si], ensure_ascii=False), encoding="utf-8")
        shard_out = shard_dir / f"shard_{mode}_{si}.jsonl"
        shard_out_files.append(shard_out)

        cmd = [
            sys.executable, script_path,
            "--config", args.config,
            "--_shard-id", str(si),
            "--_shard-out", str(shard_out),
            "--_gpu-id", gpu_id,
            "--_task-file", str(task_file),
            "--_mode", mode,
            "--_model-id", model_id,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env["TOKENIZERS_PARALLELISM"] = "false"
        print(f"  [{mode}] Shard {si} on GPU {gpu_id} "
              f"({len(shard_tasks[si])} prompts)")
        p = subprocess.Popen(cmd, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append((si, gpu_id, p))

    failed = []
    for si, gpu_id, p in procs:
        stdout, _ = p.communicate()
        text = (stdout.decode("utf-8", errors="replace") if stdout else "")
        rc = p.returncode
        lines = text.strip().splitlines()
        tail = "\n".join(lines[-5:]) if lines else "(no output)"
        print(f"  [{mode}] Shard {si} (GPU {gpu_id}) exit={rc}\n{tail}")
        if rc != 0:
            failed.append(si)
            if len(lines) > 5:
                print("...\n" + "\n".join(lines[-20:]))
    if failed:
        print(f"ERROR: {mode} shards {failed} failed!")
        sys.exit(1)

    records: List[Dict] = []
    for sf in shard_out_files:
        if not sf.exists():
            continue
        for line in sf.read_text("utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records

def load_questions(raw_file: str, limit: int) -> List[Dict[str, Any]]:
    seen: Set[int] = set()
    questions = []
    with open(raw_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            doc_id = d["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            questions.append({
                "doc_id": doc_id,
                "question": d["question"],
                "gold_answer": d["gold_answer"],
            })
    if limit > 0:
        questions = questions[:limit]
    return questions


def phase_a_draft(args, cfg, questions, out_dir, shard_dir, gpu_ids):
    """Phase A: greedy draft with 3B, collecting logprobs."""
    draft_file = out_dir / "small_drafts.jsonl"
    if draft_file.exists() and load_jsonl(draft_file):
        print("Phase A: small_drafts.jsonl exists, skipping.")
        return load_jsonl(draft_file)

    print(f"\n{'='*60}\nPhase A: Small-model greedy draft ({len(questions)} questions)\n{'='*60}")
    t0 = time.time()
    tasks = []
    for q in questions:
        tasks.append({
            "doc_id": q["doc_id"],
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "prompt": build_chat_prompt(q["question"]),
        })

    records = launch_shards(
        args, tasks, "draft", cfg["small_model"], shard_dir, gpu_ids)

    for rec in records:
        text = rec.get("generated_text", "")
        steps = split_steps(text)
        rec["steps"] = steps
        rec["n_steps"] = len(steps)
        rec["pred_answer"] = extract_boxed_answer(text)
        rec["exact_match"] = float(
            normalize_answer(rec["pred_answer"])
            == normalize_answer(rec["gold_answer"]))

        if "token_logprobs" in rec and "token_offsets" in rec:
            boundaries = step_token_boundaries(text, steps)
            rec["step_stats"] = compute_step_logprob_stats(
                rec["token_logprobs"], rec["token_offsets"], boundaries)
        else:
            rec["step_stats"] = [
                {"mean_neglogprob": 0.0, "entropy": 0.0, "n_tokens": 0}
                for _ in steps
            ]

    write_jsonl(draft_file, records)
    elapsed = time.time() - t0
    acc = sum(r["exact_match"] for r in records) / max(len(records), 1)
    print(f"Phase A done: {len(records)} drafts, acc={acc:.4f}, "
          f"time={elapsed:.1f}s")
    return records

def phase_b_trigger(args, cfg, drafts, out_dir):
    """Phase B: compute trigger scores, identify steps to upgrade."""
    trigger_file = out_dir / "trigger_decisions.jsonl"
    if trigger_file.exists() and load_jsonl(trigger_file):
        print("Phase B: trigger_decisions.jsonl exists, skipping.")
        return load_jsonl(trigger_file)

    print(f"\n{'='*60}\nPhase B: Trigger scoring\n{'='*60}")
    t0 = time.time()
    results = []
    n_triggered_total = 0
    for draft in drafts:
        steps = draft.get("steps", [])
        step_stats = draft.get("step_stats", [])
        if not steps:
            results.append({
                "doc_id": draft["doc_id"],
                "triggers": [],
                "n_triggered": 0,
            })
            continue
        triggers = decide_triggers(steps, step_stats, cfg=cfg)
        n_triggered = sum(1 for t in triggers if t["triggered"])
        n_triggered_total += n_triggered
        results.append({
            "doc_id": draft["doc_id"],
            "question": draft.get("question", ""),
            "gold_answer": draft.get("gold_answer", ""),
            "steps": steps,
            "triggers": triggers,
            "n_triggered": n_triggered,
        })

    write_jsonl(trigger_file, results)
    elapsed = time.time() - t0
    print(f"Phase B done: {n_triggered_total} steps triggered across "
          f"{len(results)} questions, time={elapsed:.1f}s")
    return results

def phase_c_large_step(args, cfg, trigger_results, out_dir, shard_dir, gpu_ids):
    """Phase C: large-model single-step generation for triggered steps."""
    interv_file = out_dir / "large_interventions.jsonl"
    if interv_file.exists() and load_jsonl(interv_file):
        print("Phase C: large_interventions.jsonl exists, skipping.")
        return load_jsonl(interv_file)

    print(f"\n{'='*60}\nPhase C: Large-model step generation (7B)\n{'='*60}")
    t0 = time.time()
    tasks = []
    for tr in trigger_results:
        steps = tr.get("steps", [])
        for trig in tr.get("triggers", []):
            if not trig["triggered"]:
                continue
            k = trig["step_idx"]
            prefix_steps = steps[:k]
            prompt = build_chat_prompt(tr["question"])
            if prefix_steps:
                prompt += "\n\n".join(prefix_steps) + "\n\n"
            tasks.append({
                "doc_id": tr["doc_id"],
                "step_idx": k,
                "question": tr["question"],
                "gold_answer": tr["gold_answer"],
                "prefix_steps": prefix_steps,
                "original_step": steps[k] if k < len(steps) else "",
                "prompt": prompt,
            })

    records = launch_shards(
        args, tasks, "large_step", cfg["large_model"], shard_dir, gpu_ids)

    for rec in records:
        rec["large_step_text"] = rec.pop("generated_text", "").strip()

    write_jsonl(interv_file, records)
    elapsed = time.time() - t0
    print(f"Phase C done: {len(records)} large-model steps, time={elapsed:.1f}s")
    return records

def phase_d_acceptance(args, cfg, interventions, out_dir):
    """Phase D: acceptance gate on large-model steps.

    In skip-acceptance mode, all interventions are accepted.
    """
    accept_file = out_dir / "acceptance_decisions.jsonl"
    if accept_file.exists() and load_jsonl(accept_file):
        print("Phase D: acceptance_decisions.jsonl exists, skipping.")
        return load_jsonl(accept_file)

    print(f"\n{'='*60}\nPhase D: Acceptance gate\n{'='*60}")
    t0 = time.time()
    results = []
    for iv in interventions:
        if args.skip_acceptance:
            compat = prefix_compatibility_score(
                iv.get("large_step_text", ""),
                iv.get("question", ""),
                iv.get("prefix_steps", []),
            )
            dec = {
                "acceptance_score": 1.0,
                "accepted": True,
                "signals": {
                    "lm_consistency": 1.0,
                    "prefix_compatibility": round(compat, 4),
                    "followability": 1.0,
                },
            }
        else:
            compat = prefix_compatibility_score(
                iv.get("large_step_text", ""),
                iv.get("question", ""),
                iv.get("prefix_steps", []),
            )
            dec = decide_acceptance(
                lm_consistency=0.75,
                prefix_compat=compat,
                followability=0.75,
                cfg=cfg,
            )
        rec = {
            "doc_id": iv["doc_id"],
            "step_idx": iv["step_idx"],
            "large_step_text": iv.get("large_step_text", ""),
            "original_step": iv.get("original_step", ""),
            "question": iv.get("question", ""),
            "gold_answer": iv.get("gold_answer", ""),
            "prefix_steps": iv.get("prefix_steps", []),
        }
        rec.update(dec)
        results.append(rec)

    write_jsonl(accept_file, results)
    n_accepted = sum(1 for r in results if r["accepted"])
    elapsed = time.time() - t0
    print(f"Phase D done: {n_accepted}/{len(results)} accepted, "
          f"time={elapsed:.1f}s")
    return results

def phase_e_continuation(
    args, cfg, drafts, acceptance_results, out_dir, shard_dir, gpu_ids,
):
    """Phase E: small-model continuation from modified prefixes."""
    cont_file = out_dir / "continuations.jsonl"
    if cont_file.exists() and load_jsonl(cont_file):
        print("Phase E: continuations.jsonl exists, skipping.")
        return load_jsonl(cont_file)

    print(f"\n{'='*60}\nPhase E: Small-model continuation (3B)\n{'='*60}")
    t0 = time.time()

    accepted_by_doc: Dict[int, List[Dict]] = {}
    for ar in acceptance_results:
        if ar["accepted"]:
            accepted_by_doc.setdefault(ar["doc_id"], []).append(ar)

    draft_by_doc = {d["doc_id"]: d for d in drafts}
    tasks = []
    for doc_id, accepted_list in accepted_by_doc.items():
        draft = draft_by_doc.get(doc_id)
        if not draft:
            continue
        steps = list(draft["steps"])
        for ar in sorted(accepted_list, key=lambda x: x["step_idx"]):
            k = ar["step_idx"]
            if k < len(steps):
                steps[k] = ar["large_step_text"]

        last_replaced = max(ar["step_idx"] for ar in accepted_list)
        prefix_steps = steps[: last_replaced + 1]
        prompt = build_prefix_prompt(draft["question"], prefix_steps)
        tasks.append({
            "doc_id": doc_id,
            "question": draft["question"],
            "gold_answer": draft["gold_answer"],
            "prefix_steps": prefix_steps,
            "n_replaced": len(accepted_list),
            "prompt": prompt,
        })

    records = launch_shards(
        args, tasks, "continuation", cfg["small_model"], shard_dir, gpu_ids)

    for rec in records:
        prefix_text = "\n\n".join(rec.get("prefix_steps", []))
        gen = rec.get("generated_text", "")
        full = prefix_text + ("\n\n" + gen if gen else "")
        rec["full_response"] = full
        rec["pred_answer"] = extract_boxed_answer(full)
        rec["exact_match"] = float(
            normalize_answer(rec["pred_answer"])
            == normalize_answer(rec["gold_answer"]))

    write_jsonl(cont_file, records)
    elapsed = time.time() - t0
    print(f"Phase E done: {len(records)} continuations, time={elapsed:.1f}s")
    return records

def phase_f_aggregate(args, cfg, drafts, continuations, acceptance_results, out_dir):
    """Phase F: merge results, compute final accuracy and token costs."""
    summary_file = out_dir / "summary.json"
    detail_file = out_dir / "per_question.jsonl"

    print(f"\n{'='*60}\nPhase F: Aggregation\n{'='*60}")

    cont_by_doc = {c["doc_id"]: c for c in continuations}
    accepted_docs = {
        ar["doc_id"] for ar in acceptance_results if ar["accepted"]
    }

    details = []
    total_correct = 0
    total_small_tokens = 0
    total_large_tokens = 0
    n_intervened = 0

    for draft in drafts:
        doc_id = draft["doc_id"]
        cont = cont_by_doc.get(doc_id)
        if cont is not None:
            pred = cont["pred_answer"]
            em = cont["exact_match"]
            small_tok = draft.get("n_tokens", 0) + cont.get("n_tokens", 0)
            large_tok = sum(
                ar.get("n_tokens", 0)
                for ar in acceptance_results
                if ar["doc_id"] == doc_id and ar["accepted"]
            )
            intervened = True
            n_intervened += 1
        else:
            pred = draft.get("pred_answer", "")
            em = draft.get("exact_match", 0.0)
            small_tok = draft.get("n_tokens", 0)
            large_tok = 0
            intervened = False

        total_correct += em
        total_small_tokens += small_tok
        total_large_tokens += large_tok
        details.append({
            "doc_id": doc_id,
            "gold_answer": draft["gold_answer"],
            "pred_answer": pred,
            "exact_match": em,
            "intervened": intervened,
            "small_tokens": small_tok,
            "large_tokens": large_tok,
            "draft_correct": draft.get("exact_match", 0.0),
        })

    n = max(len(drafts), 1)
    summary = {
        "small_model": cfg["small_model"],
        "large_model": cfg["large_model"],
        "n_questions": len(drafts),
        "keystep_sd_accuracy": round(total_correct / n, 6),
        "draft_accuracy": round(
            sum(d.get("exact_match", 0) for d in drafts) / n, 6),
        "n_intervened": n_intervened,
        "frac_intervened": round(n_intervened / n, 4),
        "total_small_tokens": total_small_tokens,
        "total_large_tokens": total_large_tokens,
        "tokens_per_question": round(
            (total_small_tokens + total_large_tokens) / n, 2),
        "small_tokens_per_question": round(total_small_tokens / n, 2),
        "large_tokens_per_question": round(total_large_tokens / n, 2),
        "trigger_config": cfg.get("trigger", {}),
        "acceptance_config": cfg.get("acceptance", {}),
    }

    Path(summary_file).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(detail_file, details)

    print(f"\nKeyStep-SD accuracy: {summary['keystep_sd_accuracy']:.4f}")
    print(f"Draft-only accuracy: {summary['draft_accuracy']:.4f}")
    print(f"Intervened: {n_intervened}/{len(drafts)} "
          f"({summary['frac_intervened']:.2%})")
    print(f"Tokens/question: {summary['tokens_per_question']:.1f} "
          f"(small={summary['small_tokens_per_question']:.1f}, "
          f"large={summary['large_tokens_per_question']:.1f})")
    print(f"Summary -> {summary_file}")
    return summary

def main():
    args = parse_args()

    if args._shard_id >= 0:
        run_shard(args)
        return

    cfg = load_config(args.config)
    gpu_ids = (args.gpus or cfg.get("gpus", "0,1,2,3,4,5,6,7")).split(",")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.raw_file, args.limit)
    print(f"Loaded {len(questions)} questions, using GPUs {gpu_ids}")

    t_start = time.time()

    drafts = phase_a_draft(args, cfg, questions, out_dir, shard_dir, gpu_ids)
    trigger_results = phase_b_trigger(args, cfg, drafts, out_dir)
    interventions = phase_c_large_step(
        args, cfg, trigger_results, out_dir, shard_dir, gpu_ids)
    acceptance_results = phase_d_acceptance(args, cfg, interventions, out_dir)
    continuations = phase_e_continuation(
        args, cfg, drafts, acceptance_results, out_dir, shard_dir, gpu_ids)
    summary = phase_f_aggregate(
        args, cfg, drafts, continuations, acceptance_results, out_dir)

    total_time = time.time() - t_start
    print(f"\nTotal pipeline time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
