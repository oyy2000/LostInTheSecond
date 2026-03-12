#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run steering-vector experiments with two apply ranges only:
1) prefix: first N tokens
2) all: from beginning to the end

This script is intentionally scoped for range comparison only.
"""

import argparse
import itertools
import json
import os
import queue
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import List


LAMBDA_STEP = 0.5
DEFAULT_STEER_LAMBDAS = [round(i * LAMBDA_STEP, 2) for i in range(-10, 11)]
DEFAULT_LAMBDAS_CSV = ",".join(str(x) for x in DEFAULT_STEER_LAMBDAS)


@dataclass
class Job:
    mode: str
    model_id: str
    layer: int
    lam: float
    prefix_n: int


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def normalize_argv_for_negative_lambdas(argv: List[str]) -> List[str]:
    """
    argparse may parse `--lambdas -1.0,-0.5,...` incorrectly because the value starts with '-'.
    Normalize it to `--lambdas=-1.0,-0.5,...`.
    """
    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--lambdas" and i + 1 < len(argv):
            nxt = argv[i + 1]
            if nxt.startswith("-"):
                out.append(f"--lambdas={nxt}")
                i += 2
                continue
        out.append(tok)
        i += 1
    return out


def run_cmd(cmd: List[str], cwd: Path, env: dict, log_file: Path) -> None:
    print("[CMD]", shlex.join(cmd))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        f.write("[CMD] " + shlex.join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        f.write(f"\n[EXIT] returncode={proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}). See log: {log_file}")


def lam_to_tag(lam: float) -> str:
    if abs(lam) < 1e-9:
        return "BASELINE"
    return f"lam{str(lam).replace('.', 'p')}"


def worker_loop(
    gpu_id: int,
    jobs_q: queue.Queue,
    python_bin: str,
    harness_dir: Path,
    args,
    records: list,
):
    while True:
        try:
            job = jobs_q.get_nowait()
        except queue.Empty:
            return

        mode_tag = "prefix" if job.mode == "prefix" else "all"
        lam_tag = lam_to_tag(job.lam)

        effective_max_token = job.prefix_n if job.mode == "prefix" else args.steer_max_token

        out_dir = (
            Path(f"{args.output_root}_N{effective_max_token}") 
            / f"{mode_tag}"
            / f"{job.model_id.split('/')[-1]}_L{job.layer}_{lam_tag}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        model_args = (
            f"pretrained={job.model_id},"
            f"dtype={args.dtype},"
            f"steer_layer={job.layer},"
            f"steer_lambda={job.lam},"
            f"steer_vec_path={args.vector_path},"
            f"steer_apply_mode={job.mode},"
            f"steer_min_token={args.steer_min_token},"
            f"steer_max_token={effective_max_token}"
        )

        cmd = [
            python_bin,
            "-m",
            "lm_eval",
            "--model",
            "steer_hf",
            "--model_args",
            model_args,
            "--tasks",
            args.tasks,
            "--device",
            "cuda:0",
            "--num_fewshot",
            str(args.num_fewshot),
            "--batch_size",
            str(args.batch_size),
            "--gen_kwargs",
            args.gen_kwargs,
            "--output_path",
            str(out_dir),
            "--log_samples",
        ]
        if args.apply_chat_template:
            cmd.append("--apply_chat_template")
        if args.limit > 0:
            cmd.extend(["--limit", str(args.limit)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        stdout_log = out_dir / "stdout.log"
        stderr_log = out_dir / "stderr.log"
        run_log = out_dir / "run.log"

        t0 = time.time()
        print(
            f"\n[GPU {gpu_id}] mode={job.mode}, layer={job.layer}, "
            f"lambda={job.lam}, prefix_n={effective_max_token}"
        )
        print(f"[GPU {gpu_id}] log -> {run_log}")

        record = {
            "mode": mode_tag,
            "model_id": job.model_id,
            "layer": job.layer,
            "lambda": job.lam,
            "prefix_n": effective_max_token,
            "is_baseline": bool(abs(job.lam) < 1e-9),
            "gpu": gpu_id,
            "status": "running",
            "returncode": None,
            "start_ts": datetime.now().isoformat(),
            "end_ts": None,
            "duration_sec": None,
            "outdir": str(out_dir),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "run_log": str(run_log),
            "cmd": shlex.join(cmd),
        }
        records.append(record)

        try:
            run_cmd(cmd, cwd=harness_dir, env=env, log_file=run_log)
            with stdout_log.open("a", encoding="utf-8") as f:
                f.write("[CMD] " + shlex.join(cmd) + "\n")
                f.write("[LOG] see run.log for merged stdout/stderr\n")
            with stderr_log.open("a", encoding="utf-8") as f:
                f.write("")
        except Exception as e:
            print(f"[GPU {gpu_id}] FAILED: {e}")
            record["status"] = "failed"
            record["returncode"] = -1
        else:
            dt = (time.time() - t0) / 60
            print(f"[GPU {gpu_id}] done in {dt:.1f} min")
            record["status"] = "done"
            record["returncode"] = 0

        end_ts = datetime.now().isoformat()
        record["end_ts"] = end_ts
        try:
            st = datetime.fromisoformat(record["start_ts"])
            ed = datetime.fromisoformat(end_ts)
            record["duration_sec"] = (ed - st).total_seconds()
        except Exception:
            pass


def main() -> None:
    argv = normalize_argv_for_negative_lambdas(sys.argv[1:])
    ap = argparse.ArgumentParser(description="Steering range experiment: prefix vs all")
    ap.add_argument("--harness-dir", default="./lm-evaluation-harness")
    ap.add_argument("--output-root", default="./runs/09_token_range_exp")
    ap.add_argument("--vector-path", default="./artifacts/vectors_first2steps/Qwen_Qwen2.5-3B-Instruct_first2steps/steering_vector.pt")

    ap.add_argument("--models", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument(
        "--modes",
        default="prefix,all",
        help="Comma-separated modes to run: prefix, all, or both.",
    )
    ap.add_argument("--layers", default="6,16,17")
    ap.add_argument("--lambdas", default=DEFAULT_LAMBDAS_CSV)

    ap.add_argument("--tasks", default="hendrycks_math_500")
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--batch-size", default="16")
    ap.add_argument("--gen-kwargs", default="max_gen_toks=2048,temperature=0,do_sample=False")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--dtype", default="float16")

    ap.add_argument("--steer-min-token", type=int, default=0)
    ap.add_argument("--steer-max-token", type=int, default=128)
    ap.add_argument(
        "--prefix-token-ns",
        default="128",
        help="Comma-separated prefix token N list for prefix mode, e.g. 32,64,128. If omitted, uses --steer-max-token.",
    )
    ap.add_argument("--apply-chat-template", action="store_true", default=True)
    ap.add_argument("--no-apply-chat-template", dest="apply_chat_template", action="store_false")

    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent.parent
    harness_dir = (root / args.harness_dir).resolve()
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    args.output_root = str(output_root)

    vector_path = Path(args.vector_path)
    if not vector_path.is_absolute():
        vector_path = (root / vector_path).resolve()
    if not vector_path.exists():
        raise FileNotFoundError(f"vector_path not found: {vector_path}")
    args.vector_path = str(vector_path)

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid_modes = {"prefix", "all"}
    if not modes:
        raise ValueError("--modes is empty. Use prefix, all, or prefix,all")
    invalid_modes = [m for m in modes if m not in valid_modes]
    if invalid_modes:
        raise ValueError(f"Invalid --modes: {invalid_modes}. Allowed: prefix,all")

    layers = parse_csv_ints(args.layers)
    lambdas = parse_csv_floats(args.lambdas)
    gpus = parse_csv_ints(args.gpus)
    prefix_token_ns = parse_csv_ints(args.prefix_token_ns) if args.prefix_token_ns else [args.steer_max_token]

    jobs = []
    for mode, model_id, layer, lam in itertools.product(modes, model_list, layers, lambdas):
        if mode == "prefix":
            for prefix_n in prefix_token_ns:
                jobs.append(Job(mode=mode, model_id=model_id, layer=layer, lam=lam, prefix_n=prefix_n))
        else:
            jobs.append(
                Job(
                    mode=mode,
                    model_id=model_id,
                    layer=layer,
                    lam=lam,
                    prefix_n=args.steer_max_token,
                )
            )

    jobs_q: queue.Queue = queue.Queue()
    for j in jobs:
        jobs_q.put(j)

    plan = {
        "vector_path": args.vector_path,
        "modes": modes,
        "models": model_list,
        "layers": layers,
        "lambdas": lambdas,
        "tasks": args.tasks,
        "num_jobs": len(jobs),
        "prefix_window": [args.steer_min_token, args.steer_max_token],
        "prefix_token_ns": prefix_token_ns,
        "gpus": gpus,
    }
    (output_root / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print("===== 09 Range Experiment =====")
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    python_bin = sys.executable
    records = []
    threads = []
    for gpu_id in gpus:
        t = Thread(target=worker_loop, args=(gpu_id, jobs_q, python_bin, harness_dir, args, records), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    runs_payload = {
        "updated_at": datetime.now().isoformat(),
        "jobs": records,
    }
    (output_root / "runs.json").write_text(json.dumps(runs_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nAll jobs done.")
    print(f"Results root: {output_root}")


if __name__ == "__main__":
    main()
