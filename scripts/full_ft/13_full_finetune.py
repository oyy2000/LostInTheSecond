#!/usr/bin/env python3
"""
Full parameter fine-tuning (no LoRA) for Qwen2.5-3B-Instruct.

Uses Adafactor (factored second moment) + gradient checkpointing to fit
a 3B model's optimizer states within a single 48GB A6000 GPU.

Memory budget (bf16 model + Adafactor + grad ckpt):
  ~6GB model + ~2GB optimizer + ~6GB grads + ~3GB activations ≈ 17GB

Usage:
    python 13_full_finetune.py \\
        --model-id Qwen/Qwen2.5-3B-Instruct \\
        --dataset-path /path/to/dataset.json \\
        --output-dir /path/to/output \\
        --learning-rate 1e-5 --num-train-epochs 3 --weight-decay 0.01
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
torch.set_float32_matmul_precision("medium")
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.optimization import Adafactor

try:
    import bitsandbytes  # noqa: F401
    DEFAULT_OPTIM = "paged_adamw_8bit"
except ImportError:
    DEFAULT_OPTIM = "adafactor"


class AdafactorTrainer(Trainer):
    """Trainer that properly configures Adafactor with a fixed learning rate."""

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        decay_params = set(self.get_decay_parameter_names(self.model))
        groups = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if n in decay_params and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if n not in decay_params and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = Adafactor(
            groups,
            lr=self.args.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
        return self.optimizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--train-ratio", type=float, default=0.95)

    p.add_argument("--num-train-epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--eval-steps", type=int, default=7)
    p.add_argument("--save-steps", type=int, default=7)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--optim", default=DEFAULT_OPTIM)

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "samples" in obj:
        return obj["samples"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Unsupported dataset format")


def build_user_prompt(tokenizer, question: str) -> str:
    question = question.strip()
    if not question:
        return ""
    messages = [{"role": "user", "content": question}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return f"Question:\n{question}\n\nAnswer:\n"


def build_dataset(records, tokenizer, max_length: int):
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    eos_token_id = tokenizer.eos_token_id

    for item in records:
        doc = item.get("doc", {})
        question = (doc.get("question") or doc.get("problem") or "").strip()
        pos = (item.get("pos_response") or "").strip()
        if not question or not pos:
            continue

        prompt_text = build_user_prompt(tokenizer, question)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        pos_ids = tokenizer(pos, add_special_tokens=False).input_ids

        if eos_token_id is not None and (not pos_ids or pos_ids[-1] != eos_token_id):
            pos_ids = pos_ids + [eos_token_id]

        input_ids = prompt_ids + pos_ids
        labels = [-100] * len(prompt_ids) + pos_ids
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

    return Dataset.from_dict(rows)


def pad_collate_fn(features, pad_token_id=0):
    """Dynamic padding to the longest sequence in the batch."""
    max_len = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        pad_len = max_len - len(f["input_ids"])
        batch["input_ids"].append(f["input_ids"] + [pad_token_id] * pad_len)
        batch["attention_mask"].append(f["attention_mask"] + [0] * pad_len)
        batch["labels"].append(f["labels"] + [-100] * pad_len)
    return {k: torch.tensor(v) for k, v in batch.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    t0 = time.time()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path)

    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")
    print(f"[Full FT] CUDA_VISIBLE_DEVICES={cuda_vis}")
    print(f"[Full FT] torch.cuda.device_count()={torch.cuda.device_count()}")
    print(f"[Full FT] Model: {args.model_id}")
    print(f"[Full FT] Optimizer: {args.optim}")
    print(f"[Full FT] LR: {args.learning_rate}, Epochs: {args.num_train_epochs}, "
          f"WD: {args.weight_decay}, Warmup: {args.warmup_ratio}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = read_records(dataset_path)
    print(f"[Full FT] Dataset: {len(records)} records from {dataset_path.name}")

    full_ds = build_dataset(records, tokenizer, args.max_length)
    if len(full_ds) < 2:
        raise ValueError(f"Not enough valid samples: {len(full_ds)}")

    split = full_ds.train_test_split(
        test_size=1 - args.train_ratio, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[Full FT] Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda:0")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Full FT] Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    import functools
    collator = functools.partial(pad_collate_fn, pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        optim=args.optim,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=0,
        torch_compile=False,
    )

    TrainerClass = AdafactorTrainer if args.optim == "adafactor" else Trainer
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()
    print(f"[Full FT] Final eval loss: {eval_result.get('eval_loss', 'N/A')}")

    best_model_dir = output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    print(f"[Full FT] Best model saved to {best_model_dir}")

    log_history = trainer.state.log_history
    eval_losses = [x["eval_loss"] for x in log_history if "eval_loss" in x]
    best_eval_loss = min(eval_losses) if eval_losses else eval_result.get("eval_loss")

    sweep_metrics = {
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
        "log_history": log_history,
        "config": {
            "model_id": args.model_id,
            "dataset_path": str(args.dataset_path),
            "method": "full_ft",
            "optim": args.optim,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds),
            "trainable_params": trainable,
            "total_params": total,
        },
        "best_eval_loss": best_eval_loss,
    }
    metrics_path = output_dir / "sweep_metrics.json"
    metrics_path.write_text(json.dumps(sweep_metrics, indent=2, default=str))

    elapsed = time.time() - t0
    print(f"[Full FT] Done in {elapsed / 60:.1f} min")
    print(f"[Full FT] Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
