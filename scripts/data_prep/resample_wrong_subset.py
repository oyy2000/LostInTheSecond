#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def load_wrong_ids(path: Path) -> List[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = data.get("ids", [])
    return [int(x) for x in ids]


def build_prompt_from_record(rec: Dict) -> str:
    # 优先使用原始 lm-eval prompt，确保采样分布最接近 baseline
    prompt = (
        (((rec.get("arguments") or {}).get("gen_args_0") or {}).get("arg_0"))
        or ""
    ).strip()
    if prompt:
        return prompt

    # fallback
    doc = rec.get("doc", {}) or {}
    q = (doc.get("problem") or doc.get("question") or "").strip()
    return f"Solve the following math problem.\nProblem: {q}\nAnswer:"


@torch.inference_mode()
def sample_many(
    model,
    tokenizer,
    prompt: str,
    n: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_return: int,
) -> List[str]:
    out: List[str] = []

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    attention_mask = enc.attention_mask.to(model.device)

    remain = n
    while remain > 0:
        k = min(batch_return, remain)
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=k,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_len = input_ids.shape[1]
        for i in range(gen_ids.shape[0]):
            text = tokenizer.decode(gen_ids[i, prompt_len:], skip_special_tokens=True)
            out.append(text.strip())

        remain -= k

    return out


def main():
    parser = argparse.ArgumentParser(description="Resample wrong subset with stochastic decoding")
    parser.add_argument("--wrong-ids", default="./artifacts/s_step2_wrong_ids.json")
    parser.add_argument("--baseline-samples", default="", help="samples_*.jsonl; if empty, auto-find in --runs-root")
    parser.add_argument("--runs-root", default="./runs/baseline_qwen25_3b_math500")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num-resamples", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-return", type=int, default=10)
    parser.add_argument("--output", default="./artifacts/wrong_subset_resamples_t1p0_top0p7.jsonl")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    baseline_path = (
        Path(args.baseline_samples)
        if args.baseline_samples
        else find_latest_baseline_samples(Path(args.runs_root))
    )
    wrong_ids = set(load_wrong_ids(Path(args.wrong_ids)))

    records = {int(r.get("doc_id", -1)): r for r in read_jsonl(baseline_path)}
    selected_ids = [x for x in wrong_ids if x in records]

    if not selected_ids:
        raise ValueError("No matching wrong sample ids found in baseline samples.")

    print(f"[Load] baseline={baseline_path}")
    print(f"[Load] wrong_ids={len(wrong_ids)}, matched={len(selected_ids)}")

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i, doc_id in enumerate(sorted(selected_ids)):
            rec = records[doc_id]
            doc = rec.get("doc", {}) or {}
            question = (doc.get("problem") or doc.get("question") or "").strip()
            answer = (doc.get("answer") or "").strip()
            prompt = build_prompt_from_record(rec)

            samples = sample_many(
                model=model,
                tokenizer=tok,
                prompt=prompt,
                n=args.num_resamples,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                batch_return=args.batch_return,
            )

            row = {
                "doc_id": int(doc_id),
                "question": question,
                "answer": answer,
                "prompt": prompt,
                "num_resamples": int(args.num_resamples),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "samples": samples,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0 or i == len(selected_ids) - 1:
                print(f"[Progress] {i + 1}/{len(selected_ids)} done")

    print(f"[Done] saved -> {out_path}")


if __name__ == "__main__":
    main()
