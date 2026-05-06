#!/usr/bin/env python3
"""
StepBack + Early-Stopping evaluation (multi-dataset consolidated table).

Pool: nd=16 drafts, ns=2 suffixes per draft.
Only suffix answers participate in majority voting (draft answers excluded).
Token cost = sum over used drafts of (prefix_tokens + suffix_tokens for consumed suffixes).
Prefix is shared across suffixes of the same draft (generated once).

Methods:
  - CoT: greedy single sample
  - SC: majority vote on N independent samples
  - StepBack (ours): majority vote on all 32 suffixes
  - AC / ESC / DSC: early-stop on independent samples
  - StepBack+AC / StepBack+ESC / StepBack+DSC (ours): early-stop on suffix sequence

PExp = number of full reasoning traces used for majority voting
Gain/1kTok = (accuracy - cot_accuracy) / (avg_tokens_per_question / 1000)

Usage:
    python scripts/17_1_interleaved_rollback_eval.py
    python scripts/17_1_interleaved_rollback_eval.py --datasets math500 gsm8k
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sc_unified import MethodConfig, make_policy
from src.prompt_templates import check_answer
from src.sweep_datasets import load_dataset_by_name

ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = ROOT / "results" / "qwen2.5_3b_instruct_budget_multisignal"
SC_BUDGET = 32
ND = 16
NS = 2

ALL_DATASETS = ["math500", "gsm8k", "amc2023", "olympiadbench"]

EARLY_STOP_METHODS = [
    ("AC", {"method": "ac", "confidence": 0.99}),
    ("ESC", {"method": "esc", "window_size": 5}),
    ("DSC", {"method": "dsc", "confidence": 0.95}),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--budget", type=int, default=SC_BUDGET)
    ap.add_argument("--nd", type=int, default=ND)
    ap.add_argument("--ns", type=int, default=NS)
    ap.add_argument("--strategy", default="prm_drop_fb_last",
                    help="Which rollback strategy to use for StepBack")
    ap.add_argument("--out-dir", default="")
    return ap.parse_args()


def load_checkpoint(ckpt_path):
    recs = [json.loads(l) for l in ckpt_path.read_text("utf-8").splitlines()
            if l.strip()]
    draft_map = {}
    sfx_map = defaultdict(list)
    sc_by_doc = defaultdict(list)
    for r in recs:
        did = r["doc_id"]
        tt = r.get("task_type", "")
        if tt == "draft":
            draft_map[(did, r["draft_idx"])] = r
        elif tt == "suffix":
            sfx_map[(did, r["draft_idx"], r["strategy"])].append(r)
        elif tt == "fullsc":
            sc_by_doc[did].append(r)
    for key in sfx_map:
        sfx_map[key].sort(key=lambda s: s["suffix_idx"])
    for did in sc_by_doc:
        sc_by_doc[did].sort(key=lambda r: r.get("sc_idx", 0))
    return draft_map, sfx_map, sc_by_doc


def compute_prefix_tokens(step_tokens_map, doc_id, draft_idx, rollback_step):
    """Compute number of tokens in the prefix up to (but not including) rollback_step.
    Uses cumulative_tokens from draft_step_tokens.json (same as 6_8 fair variant)."""
    if rollback_step <= 0:
        return 0
    st_key = f"{doc_id}|{draft_idx}"
    st_info = step_tokens_map.get(st_key)
    if not st_info:
        return 0
    cum = st_info["cumulative_tokens"]
    idx = min(rollback_step - 1, len(cum) - 1)
    return cum[idx]


def build_vanilla_sc_sequence(doc_id, sc_by_doc, draft_map, budget):
    """Vanilla SC: independent samples (fullsc or drafts). Each is a full trace."""
    if sc_by_doc.get(doc_id):
        samples = sc_by_doc[doc_id][:budget]
        return [{"answer": s["sc_answer"], "tokens": s.get("sc_tokens", 0),
                 "draft_idx": None} for s in samples]
    seq = []
    for di in range(budget):
        d = draft_map.get((doc_id, di))
        if d:
            seq.append({"answer": d.get("draft_answer", ""),
                        "tokens": d.get("draft_tokens", 0),
                        "draft_idx": None})
    return seq


def build_stepback_suffix_sequence(doc_id, draft_map, sfx_map, step_tokens_map,
                                   strategy, nd, ns):
    """Build suffix-only sequence for StepBack. Each element is a suffix answer
    with metadata for token cost: (prefix_tokens, suffix_tokens, draft_idx).
    Suffixes from the same draft share prefix cost (counted once per draft)."""
    seq = []
    for di in range(nd):
        d = draft_map.get((doc_id, di))
        if not d:
            break
        suffixes = sfx_map.get((doc_id, di, strategy), [])[:ns]
        if not suffixes:
            continue
        rollback_step = suffixes[0]["rollback_step"]
        prefix_tok = compute_prefix_tokens(step_tokens_map, doc_id, di, rollback_step)
        for si, s in enumerate(suffixes):
            seq.append({
                "answer": s["suffix_answer"],
                "suffix_tokens": s.get("suffix_tokens", 0),
                "prefix_tokens": prefix_tok,
                "draft_idx": di,
                "is_first_in_draft": (si == 0),
            })
    return seq


def run_policy_on_sequence(sequence, method_kwargs, budget):
    """Run a policy on a pre-built answer sequence.
    Returns (answer, n_used)."""
    config = MethodConfig(max_samples=budget, **method_kwargs)
    policy = make_policy(config)
    result = policy.run(generations=sequence, answer_extractor=None)
    return result.answer or "", result.samples_used


def compute_vanilla_tokens(sequence, n_used):
    """Token cost for vanilla methods: sum of per-sample tokens."""
    return sum(sequence[i].get("tokens", 0)
               for i in range(min(n_used, len(sequence))))


def compute_stepback_tokens(sequence, n_used):
    """Token cost for StepBack: prefix counted once per draft + suffix tokens.
    Each element has prefix_tokens, suffix_tokens, draft_idx, is_first_in_draft."""
    total = 0
    seen_drafts = set()
    for i in range(min(n_used, len(sequence))):
        item = sequence[i]
        di = item["draft_idx"]
        if di not in seen_drafts:
            total += item["prefix_tokens"]
            seen_drafts.add(di)
        total += item["suffix_tokens"]
    return total


def evaluate_vanilla(dataset, questions, seq_builder, method_kwargs, budget):
    """Evaluate vanilla SC/AC/ESC/DSC on independent samples."""
    nq = len(questions)
    correct = 0
    total_samples = 0
    total_tokens = 0
    for q in questions:
        seq = seq_builder(q["doc_id"])
        if not seq:
            continue
        answer, n_used = run_policy_on_sequence(seq, method_kwargs, budget)
        if check_answer(dataset, answer, q["gold_answer"]):
            correct += 1
        total_samples += n_used
        total_tokens += compute_vanilla_tokens(seq, n_used)
    return {
        "accuracy": correct / nq,
        "avg_samples": total_samples / nq,
        "avg_tokens": total_tokens / nq,
    }


def evaluate_stepback(dataset, questions, seq_builder, method_kwargs, budget):
    """Evaluate StepBack methods on suffix-only sequence with shared prefix cost."""
    nq = len(questions)
    correct = 0
    total_samples = 0
    total_tokens = 0
    for q in questions:
        seq = seq_builder(q["doc_id"])
        if not seq:
            continue
        answer, n_used = run_policy_on_sequence(seq, method_kwargs, budget)
        if check_answer(dataset, answer, q["gold_answer"]):
            correct += 1
        total_samples += n_used
        total_tokens += compute_stepback_tokens(seq, n_used)
    return {
        "accuracy": correct / nq,
        "avg_samples": total_samples / nq,
        "avg_tokens": total_tokens / nq,
    }


def evaluate_dataset(dataset, budget, strategy, nd, ns):
    """Run all methods on one dataset."""
    ckpt_path = RESULTS_BASE / dataset / "checkpoint.jsonl"
    if not ckpt_path.exists():
        print(f"  SKIP {dataset}: no checkpoint.jsonl")
        return None

    questions = load_dataset_by_name(dataset)
    draft_map, sfx_map, sc_by_doc = load_checkpoint(ckpt_path)

    # Only evaluate questions that have data in the checkpoint
    doc_ids_with_data = {did for (did, _) in draft_map.keys()}
    questions = [q for q in questions if q["doc_id"] in doc_ids_with_data]
    nq = len(questions)

    step_tokens_path = RESULTS_BASE / dataset / "draft_step_tokens.json"
    step_tokens_map = {}
    if step_tokens_path.exists():
        step_tokens_map = json.loads(step_tokens_path.read_text("utf-8"))
    else:
        print(f"  WARNING: {step_tokens_path} not found, prefix tokens will be 0")

    print(f"  {dataset}: {nq} questions, "
          f"{len(draft_map)} drafts, {len(sc_by_doc)} fullsc docs")

    results = {}

    # CoT (greedy@1)
    greedy_correct = 0
    greedy_tokens = 0
    for q in questions:
        d = draft_map.get((q["doc_id"], 0))
        if d:
            greedy_tokens += d.get("draft_tokens", 0)
            if check_answer(dataset, d.get("draft_answer", ""),
                            q["gold_answer"]):
                greedy_correct += 1
    greedy_acc = greedy_correct / nq
    greedy_avg_tok = greedy_tokens / nq
    results["CoT"] = {
        "accuracy": greedy_acc,
        "avg_samples": 1.0,
        "avg_tokens": greedy_avg_tok,
    }

    # SC (majority vote, all N independent samples)
    sc_fn = lambda did: build_vanilla_sc_sequence(
        did, sc_by_doc, draft_map, budget)
    r = evaluate_vanilla(dataset, questions, sc_fn,
                         {"method": "sc"}, budget)
    results["SC"] = r

    # StepBack (ours) — majority vote on all nd*ns suffixes (no early-stop)
    sb_fn = lambda did: build_stepback_suffix_sequence(
        did, draft_map, sfx_map, step_tokens_map, strategy, nd, ns)
    r = evaluate_stepback(dataset, questions, sb_fn,
                          {"method": "sc"}, nd * ns)
    results["StepBack (ours)"] = r

    # For each early-stop method: vanilla and StepBack+
    for es_name, es_kwargs in EARLY_STOP_METHODS:
        # Vanilla early-stop on independent samples
        van_fn = lambda did, _ek=es_kwargs: build_vanilla_sc_sequence(
            did, sc_by_doc, draft_map, budget)
        r = evaluate_vanilla(dataset, questions, van_fn,
                             es_kwargs, budget)
        results[es_name] = r

        # StepBack + early-stop on suffix sequence
        sb_es_fn = lambda did, _ek=es_kwargs: build_stepback_suffix_sequence(
            did, draft_map, sfx_map, step_tokens_map, strategy, nd, ns)
        r = evaluate_stepback(dataset, questions, sb_es_fn,
                              es_kwargs, nd * ns)
        results[f"StepBack+{es_name} (ours)"] = r

    # Compute PExp and Gain/1kTok
    for name, res in results.items():
        res["pexp"] = res["avg_samples"]
        if name == "CoT":
            res["gain_per_1k_token"] = 0.0
        else:
            avg_tok_k = res["avg_tokens"] / 1000.0
            res["gain_per_1k_token"] = (
                (res["accuracy"] - greedy_acc) / avg_tok_k
                if avg_tok_k > 0 else 0.0)

    return results


def main():
    args = parse_args()
    budget = args.budget
    strategy = args.strategy
    nd = args.nd
    ns = args.ns
    datasets = args.datasets
    out_dir = (Path(args.out_dir) if args.out_dir
               else RESULTS_BASE / "stepback_evaluation")

    print(f"LLM: Qwen2.5-3B-Instruct    N = {budget}")
    print(f"Strategy: {strategy}, nd = {nd}, ns = {ns}")
    print(f"Datasets: {datasets}\n")

    # Run evaluation for each dataset
    all_data = {}
    for ds in datasets:
        result = evaluate_dataset(ds, budget, strategy, nd, ns)
        if result is not None:
            all_data[ds] = result

    if not all_data:
        print("No datasets processed.")
        sys.exit(1)

    # Method display order
    method_order = [
        "CoT",
        "SC",
        "StepBack (ours)",
        "",  # blank separator
        "AC",
        "StepBack+AC (ours)",
        "",
        "ESC",
        "StepBack+ESC (ours)",
        "",
        "DSC",
        "StepBack+DSC (ours)",
    ]

    active_datasets = [ds for ds in datasets if ds in all_data]

    # Print consolidated table
    col_w = 22
    name_w = 22
    print("\n")
    print(f"LLM: Qwen2.5-3B-Instruct    N = {budget}, nd = {nd}, ns = {ns}")
    print()

    # Header line 1: method + dataset names
    header1 = f"{'Method':<{name_w}}"
    for ds in active_datasets:
        header1 += f"| {ds:^{col_w}} "
    print(header1)

    # Header line 2: Acc / PExp / AvgTok / Gain
    header2 = " " * name_w
    for _ in active_datasets:
        header2 += f"|{'Acc':>6} {'PExp':>5} {'Tok':>6} {'G/1kT':>6} "
    print(header2)

    # Separator
    print("-" * len(header2))

    # Data rows
    for method in method_order:
        if method == "":
            print()
            continue
        row = f"{method:<{name_w}}"
        for ds in active_datasets:
            res = all_data[ds].get(method)
            if res is None:
                row += f"|{'--':>6} {'--':>5} {'--':>6} {'--':>6} "
            else:
                acc_pct = res["accuracy"] * 100
                pexp = res["pexp"]
                avg_tok = res["avg_tokens"]
                gain = res["gain_per_1k_token"] * 100
                row += f"|{acc_pct:>6.2f} {pexp:>5.2f} {avg_tok:>6.0f} {gain:>6.3f} "
        print(row)

    # Save results
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stepback_consolidated.json"
    serializable = {}
    for ds, methods in all_data.items():
        serializable[ds] = {
            name: {key: val for key, val in res.items()}
            for name, res in methods.items()
        }
    out_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
