#!/usr/bin/env python3
"""Test: full FT with actual Trainer to verify memory fits."""
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

print("Loading model with SDPA + bf16...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
).to("cuda:0")
model.config.use_cache = False

a = torch.cuda.memory_allocated() / 1e9
print(f"Model loaded: {a:.2f}GB allocated")
print(f"attn_impl: {model.config._attn_implementation}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

seq_len = 512
dummy_data = {
    "input_ids": [list(range(100, 100 + seq_len))] * 4,
    "attention_mask": [[1] * seq_len] * 4,
    "labels": [list(range(100, 100 + seq_len))] * 4,
}
train_ds = Dataset.from_dict(dummy_data)
eval_ds = Dataset.from_dict({k: v[:1] for k, v in dummy_data.items()})

print("Creating Trainer with bf16 + gradient_checkpointing...")
training_args = TrainingArguments(
    output_dir="/tmp/test_ft_output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adafactor",
    report_to=[],
    logging_steps=1,
    max_steps=3,
    dataloader_num_workers=0,
    torch_compile=False,
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)

print("Starting training (3 steps)...")
torch.cuda.reset_peak_memory_stats()
result = trainer.train()
peak = torch.cuda.max_memory_allocated() / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"\nTraining completed!")
print(f"  Peak GPU memory: {peak:.2f}GB / {total:.2f}GB")
print(f"  Train loss: {result.training_loss:.4f}")
print("SUCCESS!")
