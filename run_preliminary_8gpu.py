import argparse
from email import parser
import itertools
import json
import os
import queue
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import List, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from steering_vectors import train_steering_vector


def parse_csv_numbers(text: str, cast):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def parse_layer_spec(spec: str, num_hidden_layers: int) -> List[int]:
    if spec.strip().lower() == "auto":
        # 早/中/晚层 + 倒数第二层
        picks = {
            max(0, num_hidden_layers // 3),
            max(0, (2 * num_hidden_layers) // 3),
            max(0, num_hidden_layers - 4),
            max(0, num_hidden_layers - 2),
        }
        return sorted([x for x in picks if x < num_hidden_layers])
    layers = parse_csv_numbers(spec, int)
    bad = [x for x in layers if x < 0 or x >= num_hidden_layers]
    if bad:
        raise ValueError(
            f"Invalid layer ids {bad}. Valid range: [0, {num_hidden_layers - 1}]"
        )
    return sorted(set(layers))


def get_score(example: dict) -> float:
    res = example.get("results", example.get("metrics", example))
    try:
        return float(res.get("exact_match", 0.0))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def make_user_prompt(tokenizer, question: str) -> str:
    msgs = [{"role": "user", "content": question.strip()}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return question.strip() + "\n"


def load_pairs(
    data_file: Path,
    tokenizer,
    max_samples: int,
    require_pos_correct: bool = True,
) -> List[Tuple[str, str]]:
    with data_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", data) if isinstance(data, dict) else data

    out = []
    for item in samples:
        if require_pos_correct and get_score(item) < 1.0:
            continue

        question = ((item.get("doc") or {}).get("question") or "").strip()
        pos = (item.get("pos_response") or "").strip()
        neg = (item.get("neg_response") or "").strip()
        if not question or not pos or not neg:
            continue

        prompt = make_user_prompt(tokenizer, question)
        out.append((prompt + pos, prompt + neg))
        if len(out) >= max_samples:
            break
    return out


def build_vector(
    model_id: str,
    data_file: Path,
    out_file: Path,
    layers: List[int],
    max_samples: int,
    read_token_index: int,
    torch_dtype: str,
):
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[torch_dtype]

    print(f"\n[Vector] loading model={model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    pairs = load_pairs(data_file, tokenizer, max_samples=max_samples, require_pos_correct=True)
    if not pairs:
        raise ValueError(f"No valid pairs found in {data_file}")

    print(f"[Vector] {data_file.name}: using {len(pairs)} pairs, layers={layers}")
    vec = train_steering_vector(
        model,
        tokenizer,
        pairs,
        layers=layers,
        read_token_index=read_token_index,
        move_to_cpu=True,
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vec, out_file)
    print(f"[Vector] saved -> {out_file}")


def run_cmd(cmd: List[str], cwd: Path, env: dict):
    print("[CMD]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


@dataclass
class EvalJob:
    vector_name: str
    vector_path: Path
    layer: int
    lam: float


def lambda_tag(lam: float) -> str:
    s = str(lam)
    s = s.replace("-", "m").replace(".", "p")
    return s


def worker_loop(
    gpu_id: int,
    jobs_q: queue.Queue,
    args,
    harness_dir: Path,
    python_bin: str,
):
    while True:
        try:
            job = jobs_q.get_nowait()
        except queue.Empty:
            return

        out_dir = (
            Path(args.output_root)
            / f"steered_{job.vector_name}_L{job.layer}_lam{lambda_tag(job.lam)}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        model_args = (
            f"pretrained={args.model_id},"
            f"dtype={args.eval_dtype},"
            f"device=cuda,"
            f"steer_layer={job.layer},"
            f"steer_lambda={job.lam},"
            f"steer_vec_path={job.vector_path}"
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
            args.task,
            "--batch_size",
            str(args.batch_size),
            "--output_path",
            str(out_dir),
            "--apply_chat_template",
            "--log_samples",
        ]
        if args.limit > 0:
            cmd += ["--limit", str(args.limit)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        print(
            f"\n[GPU {gpu_id}] vector={job.vector_name}, layer={job.layer}, lambda={job.lam}"
        )
        t0 = time.time()
        try:
            run_cmd(cmd, cwd=harness_dir, env=env)
        except Exception as e:
            print(f"[GPU {gpu_id}] FAILED: {e}")
        else:
            dt = (time.time() - t0) / 60
            print(f"[GPU {gpu_id}] done in {dt:.1f} min")


def main():
    parser = argparse.ArgumentParser(
        description="Preliminary pipeline: baseline + vector extraction + 8-GPU steering eval"
    )
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--task", default="hendrycks_math_500")
    parser.add_argument("--harness-dir", default="./lm-evaluation-harness")
    parser.add_argument("--output-root", default="./runs")
    parser.add_argument("--vectors-dir", default="./vectors_out")

    parser.add_argument("--dataset-a", default="samples_math500_ds1.json")
    parser.add_argument("--dataset-b", default="samples_math500_ds2.json")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--read-token-index", type=int, default=-1)
    parser.add_argument("--extract-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--eval-dtype", default="float16", choices=["float16", "bfloat16"])

    parser.add_argument("--layers", default="auto", help="e.g. '8,16,24,30' or 'auto'")
    parser.add_argument("--lambdas", default="0.25,0.5,1.0,2.0,-0.5,-1.0")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gen-kwargs", default="max_gen_toks=2048,temperature=0,do_sample=False", help="Generation kwargs for evaluation, e.g. 'max_gen_toks=2048,temperature=0,do_sample=False'")
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    harness_dir = (root / args.harness_dir).resolve()
    output_root = (root / args.output_root).resolve()
    vectors_dir = (root / args.vectors_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
    num_hidden_layers = int(getattr(cfg, "num_hidden_layers"))
    layers = parse_layer_spec(args.layers, num_hidden_layers)
    lambdas = parse_csv_numbers(args.lambdas, float)
    gpus = parse_csv_numbers(args.gpus, int)

    print("\n===== CONFIG =====")
    print(f"model_id      : {args.model_id}")
    print(f"task          : {args.task}")
    print(f"harness_dir   : {harness_dir}")
    print(f"layers        : {layers}")
    print(f"lambdas       : {lambdas}")
    print(f"gpus          : {gpus}")
    print(f"output_root   : {output_root}")
    print("==================\n")

    python_bin = sys.executable

    if not args.skip_baseline:
        baseline_out = output_root / "baseline_qwen25_3b_math500"
        baseline_out.mkdir(parents=True, exist_ok=True)
        baseline_model_args = f"pretrained={args.model_id},dtype={args.eval_dtype},device=cuda"
        baseline_cmd = [
            python_bin, "-m",
            "lm_eval", 
            "--model", "hf",
            "--model_args", baseline_model_args,
            "--tasks", args.task,
            "--batch_size", str(args.batch_size),
            "--output_path", str(baseline_out),
            "--gen_kwargs", str(args.gen_kwargs),        # 【控制长度】关键
            "--log_samples",
            "--apply_chat_template",
        ]

         
        if args.limit > 0:
            baseline_cmd += ["--limit", str(args.limit)]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        run_cmd(baseline_cmd, cwd=harness_dir, env=env)

    vec_a = vectors_dir / "vecA.pt"
    vec_b = vectors_dir / "vecB.pt"

    if not args.skip_extract:
        build_vector(
            model_id=args.model_id,
            data_file=(root / args.dataset_a).resolve(),
            out_file=vec_a,
            layers=layers,
            max_samples=args.max_samples,
            read_token_index=args.read_token_index,
            torch_dtype=args.extract_dtype,
        )
        build_vector(
            model_id=args.model_id,
            data_file=(root / args.dataset_b).resolve(),
            out_file=vec_b,
            layers=layers,
            max_samples=args.max_samples,
            read_token_index=args.read_token_index,
            torch_dtype=args.extract_dtype,
        )

    if not vec_a.exists() or not vec_b.exists():
        raise FileNotFoundError(
            f"Vector files missing. Expected: {vec_a} and {vec_b}. "
            "Run without --skip-extract first."
        )

    jobs = []
    for vec_name, vec_path in [("vecA", vec_a), ("vecB", vec_b)]:
        for layer, lam in itertools.product(layers, lambdas):
            jobs.append(EvalJob(vec_name, vec_path, layer, lam))

    print(f"\nTotal steering eval jobs: {len(jobs)}")
    jobs_q: queue.Queue = queue.Queue()
    for j in jobs:
        jobs_q.put(j)

    threads = []
    for gpu_id in gpus:
        t = Thread(
            target=worker_loop,
            args=(gpu_id, jobs_q, args, harness_dir, python_bin),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("\nAll preliminary jobs finished.")


if __name__ == "__main__":
    main()
