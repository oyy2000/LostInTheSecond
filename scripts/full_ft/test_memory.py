#!/usr/bin/env python3
"""Quick test: check if Full FT fits in GPU memory."""
import os
import gc
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def mem(label=""):
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  [{label}] alloc={a:.2f}GB  reserved={r:.2f}GB  total={t:.2f}GB  free={t-r:.2f}GB")


from transformers import AutoModelForCausalLM, AutoTokenizer

# Test 1: With SDPA attention
print("=" * 60)
print("Test: Loading with attn_implementation='sdpa'")
print("=" * 60)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
).to("cuda:0")
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False})
model.config.use_cache = False
model.train()
mem("after load")

print(f"  attn_impl: {model.config._attn_implementation}")
print(f"  gradient_checkpointing: {getattr(model.model, 'gradient_checkpointing', 'N/A')}")
print(f"  model.training: {model.training}")
print(f"  num_layers: {model.config.num_hidden_layers}")
print(f"  vocab_size: {model.config.vocab_size}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

for seq_len in [256, 512, 1024, 2048]:
    gc.collect()
    torch.cuda.empty_cache()
    model.zero_grad(set_to_none=True)

    dummy_ids = torch.randint(0, 1000, (1, seq_len), device="cuda:0")
    dummy_labels = dummy_ids.clone()

    torch.cuda.reset_peak_memory_stats()
    try:
        outputs = model(input_ids=dummy_ids, labels=dummy_labels)
        loss = outputs.loss
        fwd_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  seq_len={seq_len}: forward OK, loss={loss.item():.4f}")
        mem(f"fwd {seq_len}")
        print(f"  Peak during forward: {fwd_peak:.2f}GB")

        torch.cuda.reset_peak_memory_stats()
        loss.backward()
        bwd_peak = torch.cuda.max_memory_allocated() / 1e9
        mem(f"bwd {seq_len}")
        print(f"  Peak during backward: {bwd_peak:.2f}GB")

        del outputs, loss
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n  seq_len={seq_len}: OOM!")
        mem(f"OOM {seq_len}")
        print(f"  Peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
        gc.collect()
        torch.cuda.empty_cache()
        break

print("\nDone!")
