#!/usr/bin/env python3
print("step 0: script start", flush=True)

import time
t0 = time.time()

print("step 1: before import torch", flush=True)
import torch
print(f"step 2: after import torch {time.time()-t0:.2f}s", flush=True)

print("step 3: before import datasets", flush=True)
from datasets import Dataset
print(f"step 4: after import datasets {time.time()-t0:.2f}s", flush=True)

print("step 5: before import transformers", flush=True)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
print(f"step 6: after import transformers {time.time()-t0:.2f}s", flush=True)

print("step 7: before import peft", flush=True)
from peft import LoraConfig, TaskType, get_peft_model
print(f"step 8: after import peft {time.time()-t0:.2f}s", flush=True)
import argparse
import json
import os
import random
from pathlib import Path
class Timer:
    def __init__(self):
        self.last = time.time()

    def log(self, msg):
        now = time.time()
        print(f"[TIME] {msg}: {now - self.last:.2f}s")
        self.last = now

timer = Timer()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset-path", default="./artifacts/toy_data.json")
    parser.add_argument("--output-dir", default="/mnt/beegfs/youyang7/projects/LostInSecond/artifacts/lora_qwen25_3b_sft")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--max-train-samples", type=int, default=0)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    parser.add_argument("--num-train-epochs", type=float, default=3.0)
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

    return parser.parse_args()


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
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"Question:\n{question}\n\nAnswer:\n"


def build_dataset(records, tokenizer, max_length: int, max_train_samples: int = 0):
    rows = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    eos_token_id = tokenizer.eos_token_id

    count = 0
    for item in records:
        doc = item.get("doc", {})
        question = (doc.get("question") or doc.get("problem") or "").strip()
        pos = (item.get("pos_response") or "").strip()

        if not question or not pos:
            continue

        prompt_text = build_user_prompt(tokenizer, question)
        print(f"Prompt:\n{prompt_text}\nPOS:\n{pos}\n{'-'*50}")

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

        count += 1
        if max_train_samples > 0 and count >= max_train_samples:
            break

    return Dataset.from_dict(rows)


def split_dataset(ds: Dataset, train_ratio: float, seed: int):
    if len(ds) < 2:
        return ds, ds
    split = ds.train_test_split(test_size=1 - train_ratio, seed=seed)
    return split["train"], split["test"]


def main():
    print("Starting fine-tuning with LoRA...")
    args = parse_args()
    set_seed(args.seed)

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    timer.log("Tokenizer loaded")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Reading dataset...")
    records = read_records(dataset_path)
    timer.log(f"Dataset loaded ({len(records)} records)")
    
    print("Building supervised examples...")
    full_ds = build_dataset(
        records,
        tokenizer,
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
    )
    timer.log(f"Examples built ({len(full_ds)})")
    
    if len(full_ds) < 2:
        raise ValueError(f"Not enough valid samples: {len(full_ds)}")

    train_ds, eval_ds = split_dataset(full_ds, args.train_ratio, args.seed)

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

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
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
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps",
        save_strategy="steps",
        bf16=args.bf16,
        fp16=args.fp16,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
    )

    train_result = trainer.train()

    eval_result = trainer.evaluate()
    print(f"Final eval loss: {eval_result.get('eval_loss', 'N/A')}")

    sweep_metrics = {
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_result,
        "log_history": trainer.state.log_history,
        "config": {
            "model_id": args.model_id,
            "dataset_path": args.dataset_path,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds),
        },
    }
    (output_dir / "sweep_metrics.json").write_text(
        json.dumps(sweep_metrics, indent=2, default=str)
    )
    print(f"Sweep metrics saved to {output_dir / 'sweep_metrics.json'}")

    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))


if __name__ == "__main__":
    main()

    