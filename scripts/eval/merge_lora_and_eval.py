#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and save as a standalone model.
"""

import argparse
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--output-dir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {out}")
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print("Done.")


if __name__ == "__main__":
    main()
