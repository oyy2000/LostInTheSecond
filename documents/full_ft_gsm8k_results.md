# Full Fine-Tuning Sweep — Phase 1+2 GSM8K Results

**Base Model**: `Qwen/Qwen2.5-3B-Instruct`
**Method**: Full parameter fine-tuning (no LoRA)
**Optimizer**: paged_adamw_8bit (bitsandbytes)
**Task**: `gsm8k_cot_zeroshot_unified` (GSM8K zero-shot CoT, 1319 test samples)
**Training Data**: ~109 train / ~6 eval samples
**Evaluation Backend**: vLLM (bf16, greedy decoding, max_gen_toks=2048)
**Date**: 2026-03-13 09:36

## Base Model: GSM8K EM = **0.8400**

## Full Results (Ranked by GSM8K Exact Match)

| Rank | Name | Phase | GSM8K EM | Delta | Eval Loss | Best EL | Train Loss | LR | Epochs | WD | Warmup |
|------|------|-------|----------|-------|-----------|--------|------------|-----|--------|-----|--------|
| 1 | ft2_wd0005 | v2 | FAIL | N/A | 0.3565 | 0.3565 | 0.3847 | 1e-06 | 3.0 | 0.005 | 0.1 |
| 2 | ft2_wd01 | v2 | FAIL | N/A | 0.3564 | 0.3564 | 0.3846 | 1e-06 | 3.0 | 0.1 | 0.1 |
| 3 | ft2_wd01_wu02 | v2 | FAIL | N/A | 0.3566 | 0.3566 | 0.3849 | 1e-06 | 3.0 | 0.1 | 0.2 |
| 4 | ft2_wd0_wu02 | v2 | FAIL | N/A | 0.3565 | 0.3565 | 0.3850 | 1e-06 | 3.0 | 0.0 | 0.2 |
| 5 | ft2_wu0 | v2 | FAIL | N/A | 0.3568 | 0.3568 | 0.3841 | 1e-06 | 3.0 | 0.01 | 0.0 |
| 6 | ft2_wu005 | v2 | FAIL | N/A | 0.3566 | 0.3566 | 0.3843 | 1e-06 | 3.0 | 0.01 | 0.05 |
| 7 | ft2_wu02 | v2 | FAIL | N/A | 0.3565 | 0.3565 | 0.3850 | 1e-06 | 3.0 | 0.01 | 0.2 |
| 8 | ft2_wu03 | v2 | FAIL | N/A | 0.3574 | 0.3574 | 0.3852 | 1e-06 | 3.0 | 0.01 | 0.3 |

---

*Generated: 2026-03-13 09:36*
