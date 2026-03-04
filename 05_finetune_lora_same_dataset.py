#!/usr/bin/env python3
"""
LoRA SFT on the same LostInTheSecond dataset format.

Expected input JSON format:
{
  "samples": [
    {
      "doc": {"question": "...", "id": 123},
      "pos_response": "...",
      "neg_response": "...",
      "results": {"exact_match": 1.0}
    }
  ]
}

Default behavior:
- Uses `pos_response` as supervised target.
- Reuses the same dataset already produced in `artifacts/`.
- Trains only LoRA adapters and saves them to `--output-dir`.
"""

import argparse
import inspect
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:  # pragma: no cover - optional until runtime
    LoraConfig = None
    TaskType = None
    get_peft_model = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT with existing LostInTheSecond dataset")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--dataset-path",
        default="./artifacts/samples_math500_ds2_fix_step2_gpt.json",
        help="Path to dataset JSON/JSONL in LostInTheSecond pair format",
    )
    parser.add_argument("--output-dir", default="./artifacts/lora_qwen25_3b_ds2_fix_step2")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument(
        "--train-mode",
        default="standard",
        choices=["standard", "recovery_prefix_mix"],
        help="`standard`: regular SFT on pos_response. `recovery_prefix_mix`: mix normal samples with bad-prefix->repair continuation samples.",
    )
    parser.add_argument(
        "--prefix-corrupt-prob",
        type=float,
        default=0.5,
        help="For recovery_prefix_mix, probability of sampling a non-zero bad prefix length. Paper-style default is 0.5.",
    )
    parser.add_argument(
        "--recovery-max-prefix-tokens",
        type=int,
        default=100,
        help="Maximum bad-prefix token length C for recovery_prefix_mix.",
    )

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)

    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)

    parser.add_argument("--use-wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb-project", default="lostinthesecond-lora")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-tags", default="lora,ds2_fix_step2")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_records(dataset_path: Path) -> List[Dict]:
    if dataset_path.suffix.lower() == ".jsonl":
        rows = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    obj = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("samples"), list):
        return obj["samples"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Unsupported dataset format. Use JSON list, JSONL, or {samples:[...]}")


def pick_exact_match(item: Dict) -> float:
    if "exact_match" in item:
        try:
            return float(item["exact_match"])
        except Exception:
            return 0.0

    for key in ("results", "metrics", "scores"):
        block = item.get(key)
        if isinstance(block, dict) and "exact_match" in block:
            try:
                return float(block["exact_match"])
            except Exception:
                return 0.0
    return 0.0


def format_chat_example(tokenizer, question: str, answer: str) -> str:
    question = question.strip()
    answer = answer.strip()
    if not question or not answer:
        return ""

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    return f"Question:\n{question}\n\nAnswer:\n{answer}"


def build_user_prompt(tokenizer, question: str) -> str:
    question = question.strip()
    if not question:
        return ""

    messages = [{"role": "user", "content": question}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return f"Question:\n{question}\n\nAnswer:\n"


def longest_common_prefix_len(xs: List[int], ys: List[int]) -> int:
    out = 0
    for x, y in zip(xs, ys):
        if x != y:
            break
        out += 1
    return out


def decode_ids(tokenizer, token_ids: List[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def build_supervised_examples(
    records: List[Dict],
    tokenizer,
    max_train_samples: int,
    train_mode: str,
    prefix_corrupt_prob: float,
    recovery_max_prefix_tokens: int,
    seed: int,
) -> List[Dict]:
    examples: List[Dict] = []
    rng = random.Random(seed)

    for item in records:
        if pick_exact_match(item) < 1.0:
            continue

        doc = item.get("doc") or {}
        question = (doc.get("question") or doc.get("problem") or "").strip()
        pos = (item.get("pos_response") or "").strip()
        neg = (item.get("neg_response") or "").strip()

        prompt = build_user_prompt(tokenizer, question)
        if not prompt or not pos:
            continue

        examples.append(
            {
                "kind": "normal",
                "prompt": prompt,
                "assistant_prefix": "",
                "target": pos,
                "prefix_len_tokens": 0,
            }
        )

        if train_mode != "recovery_prefix_mix" or not neg:
            if max_train_samples > 0 and len(examples) >= max_train_samples:
                break
            continue

        pos_ids = tokenizer(pos, add_special_tokens=False).input_ids
        neg_ids = tokenizer(neg, add_special_tokens=False).input_ids
        shared = longest_common_prefix_len(pos_ids, neg_ids)
        capped_shared = min(shared, max(0, recovery_max_prefix_tokens))

        if capped_shared <= 0:
            if max_train_samples > 0 and len(examples) >= max_train_samples:
                break
            continue

        if rng.random() < prefix_corrupt_prob:
            k = rng.randint(1, capped_shared)
        else:
            k = 0

        target_ids = pos_ids[k:]
        if not target_ids:
            if max_train_samples > 0 and len(examples) >= max_train_samples:
                break
            continue

        examples.append(
            {
                "kind": "recovery",
                "prompt": prompt,
                "assistant_prefix": decode_ids(tokenizer, neg_ids[:k]),
                "target": decode_ids(tokenizer, target_ids),
                "prefix_len_tokens": k,
            }
        )

    if max_train_samples > 0:
        examples = examples[:max_train_samples]
    return examples


def tokenize_supervised_examples(
    examples: List[Dict],
    tokenizer,
    max_length: int,
) -> Dataset:
    rows = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "kind": [],
        "prefix_len_tokens": [],
    }

    eos_token_id = tokenizer.eos_token_id

    for item in examples:
        prompt_ids = tokenizer(item["prompt"], add_special_tokens=False).input_ids
        prefix_ids = tokenizer(item["assistant_prefix"], add_special_tokens=False).input_ids
        target_ids = tokenizer(item["target"], add_special_tokens=False).input_ids
        if eos_token_id is not None and (not target_ids or target_ids[-1] != eos_token_id):
            target_ids = target_ids + [eos_token_id]

        input_ids = prompt_ids + prefix_ids + target_ids
        labels = ([-100] * (len(prompt_ids) + len(prefix_ids))) + target_ids
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            attention_mask = attention_mask[:max_length]

        if not any(x != -100 for x in labels):
            continue

        rows["input_ids"].append(input_ids)
        rows["attention_mask"].append(attention_mask)
        rows["labels"].append(labels)
        rows["kind"].append(item["kind"])
        rows["prefix_len_tokens"].append(int(item["prefix_len_tokens"]))

    return Dataset.from_dict(rows)


def split_train_eval(texts: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if not texts:
        return [], []

    idx = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    split = int(len(idx) * train_ratio)
    split = max(1, min(split, len(idx) - 1)) if len(idx) > 1 else 1

    train_texts = [texts[i] for i in idx[:split]]
    eval_texts = [texts[i] for i in idx[split:]]
    if not eval_texts:
        eval_texts = train_texts[:1]
    return train_texts, eval_texts


def setup_wandb(args: argparse.Namespace) -> bool:
    if not args.use_wandb:
        return False

    try:
        import wandb  # type: ignore
    except Exception:
        print("[WARN] --use-wandb is enabled but wandb is not installed. Disable W&B logging.")
        return False

    run_name = args.wandb_run_name.strip() or f"lora-{Path(args.output_dir).name}"
    tags = [x.strip() for x in args.wandb_tags.split(",") if x.strip()]

    init_kwargs = {
        "project": args.wandb_project,
        "name": run_name,
        "config": vars(args),
        "tags": tags,
    }
    if args.wandb_entity.strip():
        init_kwargs["entity"] = args.wandb_entity.strip()

    wandb.init(**init_kwargs)
    return True


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if LoraConfig is None or TaskType is None or get_peft_model is None:
        raise ImportError("peft is required to run this script. Install `peft` in the active environment first.")

    dataset_path = Path(args.dataset_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = read_records(dataset_path)
    supervised_examples = build_supervised_examples(
        records,
        tokenizer,
        max_train_samples=args.max_train_samples,
        train_mode=args.train_mode,
        prefix_corrupt_prob=args.prefix_corrupt_prob,
        recovery_max_prefix_tokens=args.recovery_max_prefix_tokens,
        seed=args.seed,
    )
    if len(supervised_examples) < 2:
        raise ValueError(f"Not enough valid samples for training, got {len(supervised_examples)}")

    train_examples, eval_examples = split_train_eval(supervised_examples, train_ratio=args.train_ratio, seed=args.seed)

    train_ds = tokenize_supervised_examples(train_examples, tokenizer, args.max_length)
    eval_ds = tokenize_supervised_examples(eval_examples, tokenizer, args.max_length)

    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    wandb_enabled = setup_wandb(args)

    sig = inspect.signature(TrainingArguments.__init__)
    train_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "seed": args.seed,
    }

    if "evaluation_strategy" in sig.parameters:
        train_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in sig.parameters:
        train_kwargs["eval_strategy"] = "steps"
    elif "do_eval" in sig.parameters:
        train_kwargs["do_eval"] = True

    if "save_strategy" in sig.parameters:
        train_kwargs["save_strategy"] = "steps"

    if "report_to" in sig.parameters:
        train_kwargs["report_to"] = ["wandb"] if wandb_enabled else []

    train_args = TrainingArguments(**train_kwargs)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))

    stats = {
        "model_id": args.model_id,
        "dataset_path": str(dataset_path),
        "train_mode": args.train_mode,
        "num_records_raw": len(records),
        "num_samples_used": len(supervised_examples),
        "num_train": len(train_examples),
        "num_eval": len(eval_examples),
        "num_normal_examples": sum(1 for x in supervised_examples if x["kind"] == "normal"),
        "num_recovery_examples": sum(1 for x in supervised_examples if x["kind"] == "recovery"),
        "avg_recovery_prefix_len": (
            sum(x["prefix_len_tokens"] for x in supervised_examples if x["kind"] == "recovery")
            / max(1, sum(1 for x in supervised_examples if x["kind"] == "recovery"))
        ),
        "prefix_corrupt_prob": args.prefix_corrupt_prob,
        "recovery_max_prefix_tokens": args.recovery_max_prefix_tokens,
        "output_dir": str(output_dir),
    }
    with (output_dir / "train_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\nTraining complete.")
    print(f"Adapter saved to: {output_dir / 'final_adapter'}")
    print(f"Stats saved to: {output_dir / 'train_stats.json'}")

    if wandb_enabled:
        try:
            import wandb  # type: ignore

            wandb.log(stats)
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
