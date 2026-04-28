#!/usr/bin/env python3
"""
Phase 25-1: Alpha threshold sweep for Llama 3.2 3B.

Runs Late Rollback across a fine-grained alpha grid on all 8 datasets.
Reuses cached drafts from the Phase 15 sweep; only new suffix generations
are needed for the additional alpha values.

Usage:
    # all datasets
    python scripts/25_1_alpha_sweep_llama.py --gpus 0,1,2,3,4,5,6,7

    # single dataset
    python scripts/25_1_alpha_sweep_llama.py --dataset gsm8k --gpus 0,1,2,3
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sweep_engine import SweepConfig, run_shard, run_sweep

MODEL = "meta-llama/Llama-3.2-3B-Instruct" # "Qwen/Qwen2.5-3B-Instruct"
DATASETS = [
    "math500", "gsm8k", # "aime2024", "amc2023", "olympiadbench",
]
ALPHAS = [0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="", help="Single dataset (default: all)")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--n-drafts-list", default="4,8")
    ap.add_argument("--Ks", default="2,4")
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    ap.add_argument("--fullsc-n", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--_sid", type=int, default=-1)
    ap.add_argument("--_gpu", default="")
    ap.add_argument("--_tf", default="")
    ap.add_argument("--_out", default="")
    return ap.parse_args()


def main():
    args = parse_args()
    script = str(Path(__file__).resolve())

    if args._sid >= 0:
        run_shard(args)
        return

    datasets = [args.dataset] if args.dataset else DATASETS
    nd_list = [int(x) for x in args.n_drafts_list.split(",")]
    ks = [int(x) for x in args.Ks.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    model_short = MODEL.split("/")[-1].lower().replace("-", "_")
    root = Path(__file__).resolve().parent.parent

    for ds in datasets:
        print("\n" + "=" * 70)
        print(f"  {MODEL}  x  {ds}  (alphas={alphas})")
        print("=" * 70)
        out_dir = str(root / "results" / model_short / ds)
        cfg = SweepConfig(
            model_id=MODEL,
            dataset=ds,
            out_dir=out_dir,
            gpus=args.gpus,
            n_drafts_list=nd_list,
            Ks=ks,
            alphas=alphas,
            fullsc_n=args.fullsc_n,
        )
        run_sweep(cfg, script)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
