#!/usr/bin/env python3
"""Debug: test Qwen2.5-3B with fact_yang env."""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from vllm import LLM, SamplingParams

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def build_chat_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{question.strip()}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

llm = LLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    tensor_parallel_size=1,
    trust_remote_code=True,
    gpu_memory_utilization=0.90,
    max_model_len=2048,
    dtype="half",
)

q = "What is 2 + 3?"
prompts = [build_chat_prompt(q)]

sp = SamplingParams(temperature=0.0, max_tokens=256,
                    stop=["<|im_end|>", "<|endoftext|>"])
outputs = llm.generate(prompts, sp)

resp = outputs[0].outputs[0].text.strip()
Path("/tmp/_debug_fact_yang.txt").write_text(f"Response: {resp[:500]}\nLength: {len(resp)}\n")
print(f"Response: {resp[:200]}")
