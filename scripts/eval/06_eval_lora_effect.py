#!/usr/bin/env python3
"""
Evaluate LoRA adapter effect vs base model using lm-evaluation-harness.

It runs two evaluations on the same task:
1) base model
2) base + peft adapter

Then it extracts exact_match and writes a comparison summary JSON.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare base vs LoRA on the same eval task")
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--lora-path", default="./artifacts/lora_qwen25_3b_ds2_fix_step2/final_adapter")
    ap.add_argument("--task", default="gsm8k_cot_zeroshot_unified")
    ap.add_argument("--harness-dir", default="./lm-evaluation-harness")
    ap.add_argument("--output-root", default="./runs/lora_eval_compare")
    ap.add_argument("--batch-size", default="16")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gen-kwargs", default="max_gen_toks=2048,temperature=0,do_sample=False")
    ap.add_argument("--cuda-visible-devices", default="0")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--skip-lora", action="store_true")
    return ap.parse_args()


def run_cmd(cmd: list, cwd: Path, env: Dict[str, str]) -> None:
    print("[CMD]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def latest_results_json(run_dir: Path) -> Path:
    cand = sorted(run_dir.rglob("results_*.json"))
    if not cand:
        raise FileNotFoundError(f"No results_*.json found under: {run_dir}")
    return cand[-1]


def extract_exact_match(results_json: Path, task: str) -> Optional[float]:
    obj = json.loads(results_json.read_text(encoding="utf-8"))
    task_block: Dict[str, Any] = ((obj.get("results") or {}).get(task) or {})
    if not task_block:
        return None

    preferred_keys = [
        "exact_match,none",
        "exact_match,flexible-extract",
        "exact_match",
    ]
    for key in preferred_keys:
        if key in task_block:
            try:
                return float(task_block[key])
            except Exception:
                pass

    for key, val in task_block.items():
        if key.startswith("exact_match"):
            try:
                return float(val)
            except Exception:
                continue
    return None


def build_lm_eval_command(
    python_bin: str,
    model_backend: str,
    model_args: str,
    task: str,
    batch_size: str,
    output_path: Path,
    gen_kwargs: str,
    limit: int,
) -> list:
    cmd = [
        python_bin,
        "-m",
        "lm_eval",
        "--model",
        model_backend,
        "--model_args",
        model_args,
        "--tasks",
        task,
        "--batch_size",
        str(batch_size),
        "--gen_kwargs",
        gen_kwargs,
        "--output_path",
        str(output_path),
        "--log_samples",
        "--apply_chat_template",
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    return cmd


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parent.parent.parent
    harness_dir = (root / args.harness_dir).resolve()
    output_root = (root / args.output_root).resolve()
    lora_path = (root / args.lora_path).resolve()

    output_root.mkdir(parents=True, exist_ok=True)

    if not harness_dir.exists():
        raise FileNotFoundError(f"Harness dir not found: {harness_dir}")
    if not args.skip_lora and not lora_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {lora_path}")

    base_out = output_root / "base"
    lora_out = output_root / "lora"
    base_out.mkdir(parents=True, exist_ok=True)
    lora_out.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    base_score = None
    lora_score = None
    base_json_path = None
    lora_json_path = None

    if not args.skip_base:
        base_model_args = f"pretrained={args.model_id},dtype={args.dtype},device=cuda"
        base_cmd = build_lm_eval_command(
            python_bin=python_bin,
            model_backend="hf",
            model_args=base_model_args,
            task=args.task,
            batch_size=args.batch_size,
            output_path=base_out,
            gen_kwargs=args.gen_kwargs,
            limit=args.limit,
        )
        run_cmd(base_cmd, cwd=harness_dir, env=env)

    if not args.skip_lora:
        lora_model_args = f"pretrained={args.model_id},peft={lora_path},dtype={args.dtype},device=cuda"
        lora_cmd = build_lm_eval_command(
            python_bin=python_bin,
            model_backend="hf",
            model_args=lora_model_args,
            task=args.task,
            batch_size=args.batch_size,
            output_path=lora_out,
            gen_kwargs=args.gen_kwargs,
            limit=args.limit,
        )
        run_cmd(lora_cmd, cwd=harness_dir, env=env)

    if base_out.exists():
        try:
            base_json_path = latest_results_json(base_out)
            base_score = extract_exact_match(base_json_path, args.task)
        except Exception as e:
            print(f"[WARN] Cannot parse base results: {e}")

    if lora_out.exists():
        try:
            lora_json_path = latest_results_json(lora_out)
            lora_score = extract_exact_match(lora_json_path, args.task)
        except Exception as e:
            print(f"[WARN] Cannot parse lora results: {e}")

    delta = None
    if base_score is not None and lora_score is not None:
        delta = lora_score - base_score

    summary = {
        "task": args.task,
        "model_id": args.model_id,
        "lora_path": str(lora_path),
        "base": {
            "score_exact_match": base_score,
            "results_json": str(base_json_path) if base_json_path else None,
            "output_dir": str(base_out),
        },
        "lora": {
            "score_exact_match": lora_score,
            "results_json": str(lora_json_path) if lora_json_path else None,
            "output_dir": str(lora_out),
        },
        "delta_lora_minus_base": delta,
    }

    out_json = output_root / "comparison_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== LoRA Eval Summary =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved summary -> {out_json}")


if __name__ == "__main__":
    main()
