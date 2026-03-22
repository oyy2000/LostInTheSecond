#!/usr/bin/env python3
"""
Fine-tune LLaMA-3-8B on LEMMA-format data (alpaca: instruction/input/output).

Supports both LoRA and Full fine-tuning via --method flag.

Usage:
  # LoRA (single GPU, small-scale validation)
  python 15_finetune_lemma.py --method lora \
      --dataset-path artifacts_real/lemma_sft_wait_recompute.json \
      --output-dir /path/to/output

  # Full FT (needs more memory / multiple GPUs)
  python 15_finetune_lemma.py --method full \
      --dataset-path artifacts_real/lemma_sft_wait_recompute.json \
      --output-dir /path/to/output
"""

import argparse
import functools
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None

try:
    from transformers.optimization import Adafactor

    class AdafactorTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is not None:
                return self.optimizer
            decay_params = set(self.get_decay_parameter_names(self.model))
            groups = [
                {"params": [p for n, p in self.model.named_parameters()
                            if n in decay_params and p.requires_grad],
                 "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in self.model.named_parameters()
                            if n not in decay_params and p.requires_grad],
                 "weight_decay": 0.0},
            ]
            self.optimizer = Adafactor(
                groups, lr=self.args.learning_rate,
                scale_parameter=False, relative_step=False, warmup_init=False,
            )
            return self.optimizer
except ImportError:
    AdafactorTrainer = Trainer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["lora", "full"], default="lora")
    ap.add_argument("--model-id", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--train-ratio", type=float, default=0.95)

    ap.add_argument("--num-train-epochs", type=float, default=3.0)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=16)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--save-total-limit", type=int, default=2)

    # LoRA-specific
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules",
                     default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Full FT specific
    ap.add_argument("--optim", default="adafactor")

    return ap.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_lemma_dataset(path: Path):
    """Read LEMMA alpaca-format JSON array."""
    data = json.loads(path.read_text("utf-8"))
    if isinstance(data, dict) and "samples" in data:
        data = data["samples"]
    return data


def build_dataset(records, tokenizer, max_length: int):
    rows = {"input_ids": [], "attention_mask": [], "labels": []}
    eos_token_id = tokenizer.eos_token_id

    for item in records:
        instruction = item.get("instruction", "").strip()
        output = item.get("output", "").strip()
        if not instruction or not output:
            continue

        prompt_text = instruction + "\n"
        prompt_ids = tokenizer(prompt_text, add_special_tokens=True).input_ids
        output_ids = tokenizer(output, add_special_tokens=False).input_ids

        if eos_token_id is not None and (not output_ids or output_ids[-1] != eos_token_id):
            output_ids = output_ids + [eos_token_id]

        input_ids = prompt_ids + output_ids
        labels = [-100] * len(prompt_ids) + output_ids
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

    print(f"[FT] Method: {args.method}")
    print(f"[FT] Model: {args.model_id}")
    print(f"[FT] Dataset: {args.dataset_path}")
    print(f"[FT] LR: {args.learning_rate}, Epochs: {args.num_train_epochs}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = read_lemma_dataset(Path(args.dataset_path))
    print(f"[FT] Loaded {len(records)} records")

    full_ds = build_dataset(records, tokenizer, args.max_length)
    if len(full_ds) < 2:
        raise ValueError(f"Not enough valid samples: {len(full_ds)}")

    split = full_ds.train_test_split(test_size=1 - args.train_ratio, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[FT] Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"[FT] GPU: {gpu_name}, dtype: {torch_dtype}, bf16: {use_bf16}")

    if args.method == "lora":
        if LoraConfig is None:
            raise ImportError("peft not installed. pip install peft")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch_dtype,
            trust_remote_code=True, device_map="auto", attn_implementation="sdpa",
        )
        model.config.use_cache = False

        target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules, bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        lr = args.learning_rate
        optim = "adamw_torch"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch_dtype,
            trust_remote_code=True, attn_implementation="sdpa",
        ).to("cuda:0")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[FT] Trainable: {trainable:,} / {total:,}")

        lr = args.learning_rate if args.learning_rate != 2e-4 else 1e-5
        optim = args.optim

    collator = functools.partial(pad_collate_fn, pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
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
        bf16=use_bf16,
        fp16=not use_bf16,
        optim=optim,
        seed=args.seed,
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
    )

    TrainerClass = AdafactorTrainer if optim == "adafactor" else Trainer
    trainer = TrainerClass(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()
    print(f"[FT] Final eval loss: {eval_result.get('eval_loss', 'N/A')}")

    save_dir = output_dir / ("final_adapter" if args.method == "lora" else "best_model")
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    metrics = {
        "method": args.method,
        "model_id": args.model_id,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
        "config": vars(args),
        "elapsed_min": (time.time() - t0) / 60,
    }
    (output_dir / "sweep_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str))

    elapsed = time.time() - t0
    print(f"[FT] Done in {elapsed / 60:.1f} min -> {save_dir}")


if __name__ == "__main__":
    main()
