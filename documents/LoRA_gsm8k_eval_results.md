# GSM8K LoRA Sweep — Phase 1+2+3 Evaluation Results

**Base Model**: `Qwen/Qwen2.5-3B-Instruct`
**Task**: `gsm8k_cot_zeroshot_unified` (GSM8K zero-shot CoT, 1319 test samples)
**Training Data**: 109 train / 6 eval samples (GSM8K step-2 GPT-prefilled corrections)
**Evaluation Backend**: vLLM (bf16, greedy decoding, max_gen_toks=2048)
**Date**: 2026-03-13 09:50

## Base Model: GSM8K EM = **0.8400**

## Full Results (Ranked by GSM8K Exact Match)

| Rank | Name | Phase | GSM8K EM | Delta | Eval Loss | Best EL | Train Loss | LR | r | Alpha | Epochs | Drop | WD |
|------|------|-------|----------|-------|-----------|--------|------------|-----|---|-------|--------|------|-----|
| 1 | v3_lr1e5_r16_attn | v3 | 0.8476 | +0.0076 | 0.3555 | 0.3554 | 0.3831 | 1e-05 | 16 | 32 | 5.0 | 0.05 | 0.0 |
| 2 | v3_lr5e6_r16_all | v3 | 0.8461 | +0.0061 | 0.3565 | 0.3560 | 0.3834 | 5e-06 | 16 | 32 | 5.0 | 0.05 | 0.0 |
| 3 | v3_lr5e6_r16_attn | v3 | 0.8408 | +0.0008 | 0.3592 | 0.3575 | 0.3854 | 5e-06 | 16 | 32 | 5.0 | 0.05 | 0.0 |
| 4 | v3_lr1e6_r4_attn | v3 | 0.8400 | +0.0000 | 0.3599 | 0.3588 | 0.3861 | 1e-06 | 4 | 4 | 5.0 | 0.05 | 0.0 |
| 5 | v3_lr5e6_r4_attn | v3 | 0.8400 | +0.0000 | 0.3603 | 0.3592 | 0.3861 | 5e-06 | 4 | 4 | 5.0 | 0.05 | 0.0 |
| 6 | v3_lr1e5_r4_a1x | v3 | 0.8385 | -0.0015 | 0.3608 | 0.3594 | 0.3859 | 1e-05 | 4 | 4 | 5.0 | 0.05 | 0.0 |
| 7 | v3_lr1e5_r4_a2x_attn | v3 | 0.8378 | -0.0023 | 0.3601 | 0.3591 | 0.3858 | 1e-05 | 4 | 8 | 5.0 | 0.05 | 0.0 |
| 8 | v3_lr1e5_r4_attn | v3 | 0.8370 | -0.0030 | 0.3600 | 0.3595 | 0.3860 | 1e-05 | 4 | 4 | 5.0 | 0.05 | 0.0 |
| 9 | v3_lr1e6_r4_a1x | v3 | 0.8370 | -0.0030 | 0.3601 | 0.3596 | 0.3862 | 1e-06 | 4 | 4 | 5.0 | 0.05 | 0.0 |
| 10 | v3_lr1e6_r16_all | v3 | 0.8362 | -0.0038 | 0.3601 | 0.3596 | 0.3861 | 1e-06 | 16 | 32 | 5.0 | 0.05 | 0.0 |
| 11 | v3_lr1e5_r16_a1x_attn | v3 | 0.8362 | -0.0038 | 0.3587 | 0.3580 | 0.3851 | 1e-05 | 16 | 16 | 5.0 | 0.05 | 0.0 |
| 12 | v3_lr5e6_r4_a1x | v3 | 0.8347 | -0.0053 | 0.3597 | 0.3594 | 0.3861 | 5e-06 | 4 | 4 | 5.0 | 0.05 | 0.0 |

## Key Findings

- **Best adapter**: `v3_lr1e5_r16_attn` (EM = 0.8476, delta = +0.0076)
- **Worst adapter**: `v3_lr5e6_r4_a1x` (EM = 0.8347, delta = -0.0053)
- **3/12** adapters improved over base
- Best Phase 3: `v3_lr1e5_r16_attn` (EM = 0.8476)

---

*Generated: 2026-03-13 09:50*
*Full JSON: `/common/users/sl2148/Public/yang_ouyang/projects/LostInTheSecond/artifacts_local/lora_sweep/_gsm8k_vllm_eval/all_results.json`*
