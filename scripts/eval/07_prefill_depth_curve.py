#!/usr/bin/env python3
"""
Evaluate Step-2 shallow/deep behavior via a paper-style bad-prefix prefilling curve.

For each wrong baseline sample, prefill the first k tokens from the bad assistant
trajectory and let the same model continue generation. Then measure:
1) exact-match accuracy / recovery rate
2) repair-phrase rate

This is the reasoning analogue of the prefilling attack/diagnostic used for
alignment depth.
"""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - optional dependency at runtime
    PeftModel = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run prefilling depth curve on Step-2-wrong MATH-500 samples")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--peft-path", default="", help="Optional LoRA adapter path for augmented-model curves")
    ap.add_argument("--baseline-samples", default="", help="Path to baseline samples_*.jsonl. If empty, auto-find.")
    ap.add_argument("--runs-root", default="./runs/baseline_qwen25_3b_math500")
    ap.add_argument("--wrong-ids", default="./artifacts/s_step2_wrong_ids.json")
    ap.add_argument("--ks", default="0,5,10,20,40,80,160")
    ap.add_argument("--max-samples", type=int, default=0, help="0 means use all eligible samples")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch-size", type=int, default=1, help="Currently kept at 1 for deterministic prefix control")
    ap.add_argument("--repair-phrases", default="wait,let's recompute,previous step is wrong,step 2 seems wrong")
    ap.add_argument("--out-dir", default="./artifacts/prefill_depth_curve")
    ap.add_argument("--cuda-visible-devices", default="")
    return ap.parse_args()


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> List[str]:
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def find_latest_baseline_samples(run_root: Path) -> Path:
    cands = sorted(run_root.rglob("samples_*.jsonl"))
    if not cands:
        raise FileNotFoundError(f"No samples_*.jsonl found under {run_root}")
    return cands[-1]


def load_wrong_ids(path: Path) -> Optional[set]:
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    ids = obj.get("ids")
    if not isinstance(ids, list):
        return None
    return {int(x) for x in ids}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def latest_resp_text(rec: Dict[str, Any]) -> str:
    fr = rec.get("filtered_resps", [])
    if isinstance(fr, list) and fr:
        return (fr[0] or "").strip()
    rs = rec.get("resps", [])
    if rs and isinstance(rs[0], list) and rs[0]:
        return (rs[0][0] or "").strip()
    return ""


def load_math_utils(repo_root: Path):
    utils_path = repo_root / "lm-evaluation-harness" / "lm_eval" / "tasks" / "hendrycks_math_500" / "utils.py"
    spec = importlib.util.spec_from_file_location("hm500_utils_local", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load hendrycks_math_500 utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_eval_samples(samples_path: Path, wrong_ids: Optional[set], max_samples: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in read_jsonl(samples_path):
        doc_id = int(rec.get("doc_id", -1))
        if wrong_ids is not None and doc_id not in wrong_ids:
            continue
        if float(rec.get("exact_match", 0.0)) >= 1.0:
            continue

        doc = rec.get("doc") or {}
        prompt = (((rec.get("arguments") or {}).get("gen_args_0") or {}).get("arg_0") or "").strip()
        bad_response = latest_resp_text(rec)
        if not prompt or not bad_response or not doc:
            continue

        out.append(
            {
                "doc_id": doc_id,
                "doc": doc,
                "prompt": prompt,
                "bad_response": bad_response,
            }
        )
        if max_samples > 0 and len(out) >= max_samples:
            break
    return out


def choose_dtype(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_model_and_tokenizer(args: argparse.Namespace):
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=choose_dtype(args.dtype),
        trust_remote_code=True,
        device_map="auto",
    ).eval()

    if args.peft_path:
        if PeftModel is None:
            raise ImportError("peft is not available but --peft-path was provided")
        model = PeftModel.from_pretrained(model, args.peft_path).eval()

    return model, tokenizer


def has_repair_phrase(text: str, phrases: List[str]) -> bool:
    lower = (text or "").lower()
    return any(phrase in lower for phrase in phrases)


def run_prefill_generation(
    model,
    tokenizer,
    prompt_text: str,
    bad_response: str,
    k: int,
    max_new_tokens: int,
) -> str:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    bad_ids = tokenizer(bad_response, add_special_tokens=False).input_ids
    prefix_ids = bad_ids[: min(k, len(bad_ids))]

    if prefix_ids:
        prefix_tensor = torch.tensor([prefix_ids], dtype=prompt_ids.dtype, device=model.device)
        input_ids = torch.cat([prompt_ids, prefix_tensor], dim=1)
    else:
        input_ids = prompt_ids

    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    assistant_ids = generated[0, prompt_ids.shape[1] :]
    return tokenizer.decode(assistant_ids, skip_special_tokens=True)


def aggregate_rows(rows: List[Dict[str, Any]], ks: List[int]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for k in ks:
        k_rows = [r for r in rows if int(r["k"]) == k]
        n = len(k_rows)
        if n == 0:
            summary.append({"k": k, "n": 0, "accuracy": 0.0, "recovery_rate": 0.0, "repair_rate": 0.0})
            continue
        acc = sum(float(r["exact_match"]) for r in k_rows) / n
        repair = sum(float(r["repair_detected"]) for r in k_rows) / n
        summary.append(
            {
                "k": k,
                "n": n,
                "accuracy": acc,
                "recovery_rate": acc,
                "repair_rate": repair,
            }
        )
    return summary


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(summary: List[Dict[str, Any]], field: str, out_path: Path, title: str) -> None:
    ks = [int(x["k"]) for x in summary]
    ys = [float(x[field]) for x in summary]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, ys, marker="o")
    plt.xlabel("Prefilled bad-token count k")
    plt.ylabel(field.replace("_", " ").title())
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        print("[WARN] batch_size is currently ignored; generation runs one sample at a time.")

    repo_root = Path(__file__).resolve().parent.parent.parent
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_path = Path(args.baseline_samples) if args.baseline_samples else find_latest_baseline_samples((repo_root / args.runs_root).resolve())
    wrong_ids = load_wrong_ids((repo_root / args.wrong_ids).resolve())
    ks = parse_csv_ints(args.ks)
    repair_phrases = parse_csv_strings(args.repair_phrases)

    samples = load_eval_samples(samples_path, wrong_ids=wrong_ids, max_samples=args.max_samples)
    if not samples:
        raise ValueError("No eligible wrong samples found for prefilling evaluation.")

    math_utils = load_math_utils(repo_root)
    model, tokenizer = load_model_and_tokenizer(args)

    rows: List[Dict[str, Any]] = []
    for k in ks:
        for item in tqdm(samples, desc=f"prefill k={k}", unit="sample"):
            generated_text = run_prefill_generation(
                model=model,
                tokenizer=tokenizer,
                prompt_text=item["prompt"],
                bad_response=item["bad_response"],
                k=k,
                max_new_tokens=args.max_new_tokens,
            )
            exact_match = int(math_utils.process_results(item["doc"], [generated_text])["exact_match"])
            repair_detected = int(has_repair_phrase(generated_text, repair_phrases))
            rows.append(
                {
                    "doc_id": int(item["doc_id"]),
                    "k": int(k),
                    "exact_match": exact_match,
                    "repair_detected": repair_detected,
                    "generated_text": generated_text,
                }
            )

    summary = aggregate_rows(rows, ks)
    metadata = {
        "model_id": args.model_id,
        "peft_path": args.peft_path or None,
        "baseline_samples": str(samples_path),
        "wrong_ids": str((repo_root / args.wrong_ids).resolve()) if args.wrong_ids else None,
        "num_samples": len(samples),
        "ks": ks,
        "max_new_tokens": args.max_new_tokens,
    }

    (out_dir / "prefill_curve_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "prefill_curve_summary.json").write_text(
        json.dumps({"summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "prefill_curve_samples.json").write_text(
        json.dumps({"samples": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_csv(out_dir / "prefill_curve_summary.csv", summary)
    save_csv(out_dir / "prefill_curve_samples.csv", rows)
    plot_curve(summary, "accuracy", out_dir / "prefill_accuracy_vs_k.png", "Prefill Depth Curve: Accuracy vs k")
    plot_curve(summary, "repair_rate", out_dir / "prefill_repair_rate_vs_k.png", "Prefill Depth Curve: Repair Rate vs k")

    print(json.dumps({"metadata": metadata, "summary": summary}, indent=2, ensure_ascii=False))
    print(f"Saved prefilling depth artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
