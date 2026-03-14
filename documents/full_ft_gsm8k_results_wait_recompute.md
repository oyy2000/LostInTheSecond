# Full Fine-Tuning Sweep — Wait+Recompute Dataset Results

**Base Model**: `Qwen/Qwen2.5-3B-Instruct`
**Method**: Full parameter fine-tuning (no LoRA)
**Optimizer**: paged_adamw_8bit (bitsandbytes)
**Task**: `gsm8k_cot_zeroshot_unified` (GSM8K zero-shot CoT, 1319 test samples)
**Training Data**: Wait+Recompute prefill dataset (pos = step1 + old_step2 + Wait... + corrected_step2 + tail)
**Evaluation Backend**: vLLM (bf16, greedy decoding, max_gen_toks=2048)
**Date**: 2026-03-13 16:40

## Base Model: GSM8K EM = **0.8400**

## Full Results (Ranked by GSM8K Exact Match)

| Rank | Name | Phase | GSM8K EM | Delta | Eval Loss | Best EL | Train Loss | LR | Epochs | WD | Warmup |
|------|------|-------|----------|-------|-----------|--------|------------|-----|--------|-----|--------|
| 1 | ft2_wd0 | v2 | 0.8378 | -0.0023 | 0.4515 | 0.4515 | 0.5149 | 1e-06 | 3.0 | 0.0 | 0.1 |
| 2 | ft2_wd0005 | v2 | 0.8378 | -0.0023 | 0.4515 | 0.4515 | 0.5149 | 1e-06 | 3.0 | 0.005 | 0.1 |
| 3 | ft2_wu0 | v2 | 0.8378 | -0.0023 | 0.4484 | 0.4484 | 0.5142 | 1e-06 | 3.0 | 0.01 | 0.0 |
| 4 | ft2_wu005 | v2 | 0.8378 | -0.0023 | 0.4515 | 0.4515 | 0.5149 | 1e-06 | 3.0 | 0.01 | 0.05 |
| 5 | ft2_wd01_wu02 | v2 | 0.8370 | -0.0030 | 0.4508 | 0.4508 | 0.5153 | 1e-06 | 3.0 | 0.1 | 0.2 |
| 6 | ft2_wd0_wu02 | v2 | 0.8355 | -0.0045 | 0.4498 | 0.4498 | 0.5154 | 1e-06 | 3.0 | 0.0 | 0.2 |
| 7 | ft2_wu02 | v2 | 0.8355 | -0.0045 | 0.4498 | 0.4498 | 0.5154 | 1e-06 | 3.0 | 0.01 | 0.2 |
| 8 | ft2_wd005 | v2 | 0.8340 | -0.0061 | 0.4495 | 0.4495 | 0.5151 | 1e-06 | 3.0 | 0.05 | 0.1 |
| 9 | ft2_wu03 | v2 | 0.8332 | -0.0068 | 0.4506 | 0.4506 | 0.5155 | 1e-06 | 3.0 | 0.01 | 0.3 |

## Key Findings

- **Best full FT**: `ft2_wd0` (EM = 0.8378, delta = -0.0023)
- **Worst full FT**: `ft2_wu03` (EM = 0.8332, delta = -0.0068)
- **0/9** experiments improved over base

---

*Generated: 2026-03-13 16:40*
