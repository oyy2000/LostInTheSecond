#!/usr/bin/env python3
"""
Paired prediction analysis: compare base model vs best adapters.

Joins per-sample predictions from base and adapter models, categorizes
each test sample into:
  (a) both correct
  (b) both wrong
  (c) base wrong -> adapter correct  (fixed)
  (d) base correct -> adapter wrong  (broken)

Outputs:
  - documents/paired_prediction_analysis.md  (summary)
  - artifacts/paired_analysis_details.jsonl  (detailed category c/d samples)

Usage:
    python 21_paired_prediction_analysis.py
    python 21_paired_prediction_analysis.py --adapters attn_only,v2_lr1e4_r4_a1x,lr_1e-5
    python 21_paired_prediction_analysis.py --include-ft --ft-models ft_lr5e6,ft_lr1e6
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

BEEGFS_ARTIFACTS = Path("/mnt/beegfs/youyang7/projects/LostInSecond/artifacts")
LORA_EVAL_ROOT = BEEGFS_ARTIFACTS / "lora_sweep" / "_gsm8k_vllm_eval"
FT_EVAL_ROOT = BEEGFS_ARTIFACTS / "full_ft_sweep" / "_gsm8k_vllm_eval"

LOCAL_LORA_EVAL = PROJECT_ROOT / "artifacts" / "lora_sweep" / "_gsm8k_vllm_eval"
LOCAL_FT_EVAL = PROJECT_ROOT / "artifacts" / "full_ft_sweep" / "_gsm8k_vllm_eval"

TRAINING_DATA = BEEGFS_ARTIFACTS / "samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json"
LOCAL_TRAINING_DATA = ARTIFACTS_DIR / "samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json"


def find_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def find_samples_file(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.rglob("samples_gsm8k_cot_zeroshot_unified_*.jsonl"))
    return candidates[-1] if candidates else None


def load_samples(samples_path: Path) -> Dict[int, Dict[str, Any]]:
    """Load full sample records keyed by doc_id."""
    records = {}
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            if doc_id is not None:
                records[doc_id] = obj
    return records


def load_training_questions() -> set:
    """Load question IDs from training data to check overlap."""
    train_path = find_path([TRAINING_DATA, LOCAL_TRAINING_DATA])
    if train_path is None:
        return set()

    data = json.loads(train_path.read_text())
    samples = data.get("samples", data) if isinstance(data, dict) else data
    train_ids = set()
    for s in samples:
        doc = s.get("doc", {})
        qid = doc.get("id")
        if qid is not None:
            train_ids.add(qid)
    return train_ids


def categorize(base_samples: Dict[int, Dict],
               model_samples: Dict[int, Dict]) -> Dict[str, List[Dict]]:
    common_ids = sorted(set(base_samples) & set(model_samples))

    categories = {
        "both_correct": [],
        "both_wrong": [],
        "fixed": [],
        "broken": [],
    }

    for doc_id in common_ids:
        base_rec = base_samples[doc_id]
        model_rec = model_samples[doc_id]

        base_em = base_rec.get("exact_match", 0)
        model_em = model_rec.get("exact_match", 0)
        base_ok = float(base_em) >= 0.5
        model_ok = float(model_em) >= 0.5

        doc = base_rec.get("doc", {})
        question = doc.get("question", "")
        answer = doc.get("answer", "")

        base_resp = ""
        if base_rec.get("resps"):
            base_resp = base_rec["resps"][0][0] if base_rec["resps"][0] else ""

        model_resp = ""
        if model_rec.get("resps"):
            model_resp = model_rec["resps"][0][0] if model_rec["resps"][0] else ""

        entry = {
            "doc_id": doc_id,
            "question": question[:200],
            "gold_answer": answer[-50:] if answer else "",
            "base_correct": base_ok,
            "model_correct": model_ok,
            "base_response_snippet": base_resp[-200:],
            "model_response_snippet": model_resp[-200:],
        }

        if base_ok and model_ok:
            categories["both_correct"].append(entry)
        elif not base_ok and not model_ok:
            categories["both_wrong"].append(entry)
        elif not base_ok and model_ok:
            categories["fixed"].append(entry)
        else:
            categories["broken"].append(entry)

    return categories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="attn_only,v2_lr1e4_r4_a1x,lr_1e-5",
                    help="Comma-separated LoRA adapter names to analyze")
    ap.add_argument("--include-ft", action="store_true",
                    help="Also analyze full FT models")
    ap.add_argument("--ft-models", default="ft_lr5e6,ft_lr1e6",
                    help="Comma-separated full FT model names")
    args = ap.parse_args()

    lora_root = find_path([LORA_EVAL_ROOT, LOCAL_LORA_EVAL])
    ft_root = find_path([FT_EVAL_ROOT, LOCAL_FT_EVAL])

    # Load base
    base_dir = None
    for root in [lora_root, ft_root]:
        if root and (root / "base").exists():
            base_dir = root / "base"
            break

    if base_dir is None:
        print("ERROR: No base model evaluation directory found.")
        return

    base_samples_file = find_samples_file(base_dir)
    if base_samples_file is None:
        print(f"ERROR: No samples file in {base_dir}")
        return

    print(f"Loading base predictions: {base_samples_file}")
    base_samples = load_samples(base_samples_file)
    print(f"  {len(base_samples)} samples loaded")

    base_em = sum(1 for s in base_samples.values()
                  if float(s.get("exact_match", 0)) >= 0.5) / len(base_samples)

    train_ids = load_training_questions()
    print(f"Training set question IDs: {len(train_ids)}")

    models_to_analyze = []

    adapter_names = [n.strip() for n in args.adapters.split(",") if n.strip()]
    for name in adapter_names:
        if lora_root and (lora_root / name).exists():
            sf = find_samples_file(lora_root / name)
            if sf:
                models_to_analyze.append({"name": name, "method": "lora",
                                          "samples_path": sf})

    if args.include_ft:
        ft_names = [n.strip() for n in args.ft_models.split(",") if n.strip()]
        for name in ft_names:
            if ft_root and (ft_root / name).exists():
                sf = find_samples_file(ft_root / name)
                if sf:
                    models_to_analyze.append({"name": name, "method": "full_ft",
                                              "samples_path": sf})

    if not models_to_analyze:
        print("ERROR: No adapter/model predictions found.")
        return

    # Analyze each model
    all_results = []
    all_details = []

    for m in models_to_analyze:
        print(f"\nAnalyzing: {m['name']} ({m['method']})")
        model_samples = load_samples(m["samples_path"])
        categories = categorize(base_samples, model_samples)

        n_total = sum(len(v) for v in categories.values())
        model_em = (len(categories["both_correct"]) + len(categories["fixed"])) / n_total

        # Cross-reference fixed/broken with training data
        fixed_in_train = sum(
            1 for e in categories["fixed"]
            if e["doc_id"] in train_ids
        )
        broken_in_train = sum(
            1 for e in categories["broken"]
            if e["doc_id"] in train_ids
        )

        result = {
            "name": m["name"],
            "method": m["method"],
            "total": n_total,
            "both_correct": len(categories["both_correct"]),
            "both_wrong": len(categories["both_wrong"]),
            "fixed": len(categories["fixed"]),
            "broken": len(categories["broken"]),
            "net_improvement": len(categories["fixed"]) - len(categories["broken"]),
            "base_em": base_em,
            "model_em": model_em,
            "fixed_in_train": fixed_in_train,
            "broken_in_train": broken_in_train,
        }
        all_results.append(result)

        print(f"  Both correct:  {result['both_correct']}")
        print(f"  Both wrong:    {result['both_wrong']}")
        print(f"  Fixed:         {result['fixed']} "
              f"({fixed_in_train} overlap with train)")
        print(f"  Broken:        {result['broken']} "
              f"({broken_in_train} overlap with train)")
        print(f"  Net:           {result['net_improvement']:+d}")

        for entry in categories["fixed"]:
            entry["model_name"] = m["name"]
            entry["category"] = "fixed"
            entry["in_training_set"] = entry["doc_id"] in train_ids
            all_details.append(entry)

        for entry in categories["broken"]:
            entry["model_name"] = m["name"]
            entry["category"] = "broken"
            entry["in_training_set"] = entry["doc_id"] in train_ids
            all_details.append(entry)

    # Write details JSONL
    details_path = ARTIFACTS_DIR / "paired_analysis_details.jsonl"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with open(details_path, "w") as f:
        for entry in all_details:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nDetailed results: {details_path} ({len(all_details)} entries)")

    # Generate markdown report
    lines = [
        "# Paired Prediction Analysis",
        "",
        f"**Base Model EM**: {base_em:.4f} ({len(base_samples)} test samples)",
        f"**Training data overlap**: {len(train_ids)} question IDs in train set",
        f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary Table",
        "",
        "| Model | Method | Both OK | Both Wrong | Fixed | Broken | Net | "
        "Model EM | Fix in Train | Break in Train |",
        "|-------|--------|---------|------------|-------|--------|-----|"
        "---------|--------------|----------------|",
    ]

    for r in all_results:
        lines.append(
            f"| {r['name']} | {r['method']} | {r['both_correct']} | "
            f"{r['both_wrong']} | {r['fixed']} | {r['broken']} | "
            f"{r['net_improvement']:+d} | {r['model_em']:.4f} | "
            f"{r['fixed_in_train']} | {r['broken_in_train']} |"
        )

    lines.extend(["", "## Interpretation", ""])

    for r in all_results:
        lines.extend([
            f"### {r['name']} ({r['method']})",
            "",
            f"- Fixed {r['fixed']} problems the base model got wrong",
            f"- Broke {r['broken']} problems the base model got right",
            f"- Net change: {r['net_improvement']:+d} correct answers",
            f"- {r['fixed_in_train']}/{r['fixed']} fixed problems "
            f"overlap with training set questions",
            f"- {r['broken_in_train']}/{r['broken']} broken problems "
            f"overlap with training set questions",
            "",
        ])

    lines.extend([
        "## Detailed Samples",
        "",
        f"See `artifacts/paired_analysis_details.jsonl` for full response "
        f"snippets of all fixed and broken samples.",
        "",
        "---",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*",
    ])

    report_path = DOCUMENTS_DIR / "paired_prediction_analysis.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
