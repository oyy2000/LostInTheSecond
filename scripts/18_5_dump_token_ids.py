#!/usr/bin/env python3
"""Dump first prompt token IDs from both harness and our pipeline, compare."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

HARNESS_SAMPLES = ROOT / "results/harness_math500_vllm/meta-llama__Llama-3.2-3B-Instruct/samples_hendrycks_math_500_2026-04-26T14-42-04.294052.jsonl"
OUR_SAMPLES = ROOT / "results/math500_vllm_4096_llama_3.2_3b_instruct.jsonl"

MODEL = "meta-llama/Llama-3.2-3B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# --- Harness prompt (from saved sample) ---
h_line = json.loads(HARNESS_SAMPLES.read_text("utf-8").splitlines()[0])
h_prompt = h_line["arguments"]["gen_args_0"]["arg_0"]

# --- Our prompt (from saved sample) ---
o_line = json.loads(OUR_SAMPLES.read_text("utf-8").splitlines()[0])
o_prompt = o_line["prompt"]

print("=== HARNESS PROMPT (first 300 chars) ===")
print(repr(h_prompt[:300]))
print()
print("=== OUR PROMPT (first 300 chars) ===")
print(repr(o_prompt[:300]))
print()

# Check string equality
if h_prompt == o_prompt:
    print("STRINGS: IDENTICAL")
else:
    print("STRINGS: DIFFERENT")
    # Find first difference
    for i, (a, b) in enumerate(zip(h_prompt, o_prompt)):
        if a != b:
            print(f"  First diff at char {i}: harness={repr(a)} ours={repr(b)}")
            print(f"  harness context: ...{repr(h_prompt[max(0,i-20):i+20])}...")
            print(f"  ours context:    ...{repr(o_prompt[max(0,i-20):i+20])}...")
            break
    if len(h_prompt) != len(o_prompt):
        print(f"  Length: harness={len(h_prompt)} ours={len(o_prompt)}")
print()

# Tokenize both with add_special_tokens=False (harness style)
h_ids = tok.encode(h_prompt, add_special_tokens=False)
o_ids = tok.encode(o_prompt, add_special_tokens=False)

# Also tokenize ours WITH add_special_tokens=True (what vllm.generate(str) does)
o_ids_with_special = tok.encode(o_prompt, add_special_tokens=True)

print(f"=== TOKEN IDS (first 30) ===")
print(f"Harness (add_special=False):  {h_ids[:30]}")
print(f"Ours    (add_special=False):  {o_ids[:30]}")
print(f"Ours    (add_special=True):   {o_ids_with_special[:30]}")
print()

print(f"Lengths: harness={len(h_ids)}  ours_nospecial={len(o_ids)}  ours_special={len(o_ids_with_special)}")
print()

# Decode first few tokens to see what they are
print("=== FIRST 10 TOKENS DECODED ===")
print("Harness:")
for i, tid in enumerate(h_ids[:10]):
    print(f"  [{i}] id={tid:6d}  -> {repr(tok.decode([tid]))}")
print("Ours (add_special=False):")
for i, tid in enumerate(o_ids[:10]):
    print(f"  [{i}] id={tid:6d}  -> {repr(tok.decode([tid]))}")
print("Ours (add_special=True):")
for i, tid in enumerate(o_ids_with_special[:10]):
    print(f"  [{i}] id={tid:6d}  -> {repr(tok.decode([tid]))}")
print()

# Compare harness vs ours (no special)
if h_ids == o_ids:
    print("TOKEN IDS (no special): IDENTICAL")
else:
    print("TOKEN IDS (no special): DIFFERENT")
    for i, (a, b) in enumerate(zip(h_ids, o_ids)):
        if a != b:
            print(f"  First diff at pos {i}: harness={a} ({repr(tok.decode([a]))}) vs ours={b} ({repr(tok.decode([b]))})")
            break
