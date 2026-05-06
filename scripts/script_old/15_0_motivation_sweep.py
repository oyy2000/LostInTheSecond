#!/usr/bin/env python3
"""
Phase 1: Motivation validation sweep.

Runs Late Rollback + Full SC on motivation datasets (HotpotQA, MATH500, GSM8K)
across motivation models (Llama 3.2 3B, Qwen 2.5 3B, Qwen 2.5 7B).

K = per-draft sample budget. Each draft produces K answers:
  1 greedy + (K-1) suffix continuations. K=2 means 1 extra suffix per draft.

Usage:
    # single dataset x model
    python scripts/15_0_motivation_sweep.py \\
        --dataset gsm8k --model-id Qwen/Qwen2.5-3B-Instruct --gpus 0,1,2,3

    # all motivation combos
    python scripts/15_0_motivation_sweep.py --batch --gpus 0,1,2,3,4,5,6,7

    # with config override
    python scripts/15_0_motivation_sweep.py --config configs/sweep_default.yaml --dataset math500
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sweep_engine import SweepConfig, run_shard, run_sweep

MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
DATASETS = ["hotpotqa", "math500", "gsm8k"]


def parse_args():
    ap = argparse.ArgumentParser(description="Motivation validation sweep")
    ap.add_argument("--config", default="")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--n-sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-drafts-list", default="")
    ap.add_argument("--alphas", default="")
    ap.add_argument("--Ks", default="")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--fullsc-n", type=int, default=None, help="Fixed Full SC sample count (default 40)")
    ap.add_argument("--batch", action="store_true", help="Run all motivation combos")
    # internal shard args
    ap.add_argument("--_sid", type=int, default=-1)
    ap.add_argument("--_gpu", default="")
    ap.add_argument("--_tf", default="")
    ap.add_argument("--_out", default="")
    return ap.parse_args()


def build_config(args) -> SweepConfig:
    cfg = SweepConfig()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            d = yaml.safe_load(f) or {}
        for k, v in d.items():
            k_py = k.replace("-", "_")
            if hasattr(cfg, k_py):
                setattr(cfg, k_py, v)
    if args.model_id:
        cfg.model_id = args.model_id
    if args.dataset:
        cfg.dataset = args.dataset
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.gpus:
        cfg.gpus = args.gpus
    if args.n_sample is not None:
        cfg.n_sample = args.n_sample
    if args.seed is not None:
        cfg.seed = args.seed
    if args.n_drafts_list:
        cfg.n_drafts_list = [int(x) for x in args.n_drafts_list.split(",")]
    if args.alphas:
        cfg.alphas = [float(x) for x in args.alphas.split(",")]
    if args.Ks:
        cfg.Ks = [int(x) for x in args.Ks.split(",")]
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.max_tokens is not None:
        cfg.max_tokens = args.max_tokens
    if args.gpu_memory_utilization is not None:
        cfg.gpu_memory_utilization = args.gpu_memory_utilization
    if args.max_model_len is not None:
        cfg.max_model_len = args.max_model_len
    if args.fullsc_n is not None:
        cfg.fullsc_n = args.fullsc_n
    return cfg


def main():
    args = parse_args()
    script = str(Path(__file__).resolve())

    if args._sid >= 0:
        run_shard(args)
        return

    if args.batch:
        for model in MODELS:
            for ds in DATASETS:
                print("\n" + "=" * 70)
                print(f"  {model}  x  {ds}")
                print("=" * 70)
                args.model_id = model
                args.dataset = ds
                args.out_dir = ""
                cfg = build_config(args)
                run_sweep(cfg, script)
        return

    cfg = build_config(args)
    if not cfg.dataset:
        print("ERROR: --dataset required (or use --batch)")
        sys.exit(1)
    run_sweep(cfg, script)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
