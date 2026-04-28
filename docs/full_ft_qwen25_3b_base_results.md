# Full FT Sweep on Qwen/Qwen2.5-3B (Base) -- GPT-Prefill

**Model**: `Qwen/Qwen2.5-3B` (base, non-instruct)
**Dataset**: GPT-Prefill
**Method**: Full parameter fine-tuning (Adafactor optimizer)
**Eval Tasks**: GSM8K (1319 test) + MATH-500 (500 test)
**Date**: 2026-03-14 21:42

## Baseline: GSM8K EM = **0.731614859742229** | MATH-500 EM = **0.496**

## Full Results (Ranked by GSM8K EM)

| Rank | Name | GSM8K EM | GSM8K Delta | MATH-500 EM | MATH Delta | LR | Epochs | WD | Warmup |
|------|------|----------|-------------|-------------|------------|----|--------|-----|--------|
| 1 | ft2_wd01 | 0.7316 | +0.0000 | 0.5000 | +0.0040 | 1e-06 | 3.0 | 0.1 | 0.1 |
| 2 | ft2_wd0 | 0.7293 | -0.0023 | 0.4980 | +0.0020 | 1e-06 | 3.0 | 0.0 | 0.1 |
| 3 | ft2_wd0005 | 0.7293 | -0.0023 | 0.4980 | +0.0020 | 1e-06 | 3.0 | 0.005 | 0.1 |
| 4 | ft2_wu0 | 0.7293 | -0.0023 | 0.4960 | +0.0000 | 1e-06 | 3.0 | 0.01 | 0.0 |
| 5 | ft2_wu03 | 0.7293 | -0.0023 | 0.5120 | +0.0160 | 1e-06 | 3.0 | 0.01 | 0.3 |
| 6 | ft2_wd01_wu02 | 0.7286 | -0.0030 | 0.4900 | -0.0060 | 1e-06 | 3.0 | 0.1 | 0.2 |
| 7 | ft2_wd0_wu02 | 0.7286 | -0.0030 | 0.4980 | +0.0020 | 1e-06 | 3.0 | 0.0 | 0.2 |
| 8 | ft2_wu02 | 0.7286 | -0.0030 | 0.4980 | +0.0020 | 1e-06 | 3.0 | 0.01 | 0.2 |
| 9 | ft2_wd005 | 0.7278 | -0.0038 | 0.4940 | -0.0020 | 1e-06 | 3.0 | 0.05 | 0.1 |
| 10 | ft2_wu005 | 0.7263 | -0.0053 | 0.4960 | +0.0000 | 1e-06 | 3.0 | 0.01 | 0.05 |

## Key Findings

- **Best full FT**: `ft2_wd01` (GSM8K EM = 0.7316, delta = +0.0000)
- **0/10** experiments improved over base on GSM8K

---

*Generated: 2026-03-14 21:42*
