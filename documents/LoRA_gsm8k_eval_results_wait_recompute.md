# GSM8K LoRA Sweep — Wait+Recompute Dataset Results

**Base Model**: `Qwen/Qwen2.5-3B-Instruct`
**Task**: `gsm8k_cot_zeroshot_unified` (GSM8K zero-shot CoT, 1319 test samples)
**Training Data**: Wait+Recompute prefill dataset (pos = step1 + old_step2 + Wait... + corrected_step2 + tail)
**Evaluation Backend**: vLLM (bf16, greedy decoding, max_gen_toks=2048)
**Date**: 2026-03-13 16:40

## Base Model: GSM8K EM = **0.8400**

## Full Results (Ranked by GSM8K Exact Match)

| Rank | Name | Phase | GSM8K EM | Delta | Eval Loss | Best EL | Train Loss | LR | r | Alpha | Epochs | Drop | WD |
|------|------|-------|----------|-------|-----------|--------|------------|-----|---|-------|--------|------|-----|
| 1 | v3_lr5e6_r16_all | v3 | 0.8506 | +0.0106 | 0.4502 | 0.4502 | 0.5206 | 5e-06 | 16 | 32 | 5.0 | 0.05 | 0.0 |
| 2 | v3_lr1e5_r16_attn | v3 | 0.8446 | +0.0045 | 0.4513 | 0.4513 | 0.5205 | 1e-05 | 16 | 32 | 5.0 | 0.05 | 0.0 |
| 3 | v3_lr1e6_r4_a1x | v3 | 0.8438 | +0.0038 | 0.4549 | 0.4545 | 0.5221 | 1e-06 | 4 | 4 | 5.0 | 0.05 | 0.0 |
| 4 | v3_lr1e5_r4_attn | v3 | 0.8408 | +0.0008 | 0.4551 | 0.4549 | 0.5221 | 1e-05 | 4 | 4 | 5.0 | 0.05 | 0.0 |
| 5 | v3_lr1e6_r16_attn | v3 | 0.8408 | +0.0008 | 0.4540 | 0.4540 | 0.5220 | 1e-06 | 16 | 32 | 5.0 | 0.05 | 0.0 |
| 6 | v3_lr5e6_r4_a1x | v3 | 0.8400 | +0.0000 | 0.4553 | 0.4543 | 0.5222 | 5e-06 | 4 | 4 | 5.0 | 0.05 | 0.0 |
| 7 | v3_lr5e6_r4_attn | v3 | 0.8400 | +0.0000 | 0.4545 | 0.4545 | 0.5222 | 5e-06 | 4 | 4 | 5.0 | 0.05 | 0.0 |
| 8 | v3_lr1e6_r16_all | v3 | 0.8393 | -0.0008 | 0.4541 | 0.4541 | 0.5219 | 1e-06 | 16 | 32 | 5.0 | 0.05 | 0.0 |
| 9 | v3_lr1e6_r4_attn | v3 | 0.8385 | -0.0015 | 0.4553 | 0.4546 | 0.5222 | 1e-06 | 4 | 4 | 5.0 | 0.05 | 0.0 |
| 10 | v3_lr1e5_r16_a1x_attn | v3 | 0.8378 | -0.0023 | 0.4535 | 0.4535 | 0.5218 | 1e-05 | 16 | 16 | 5.0 | 0.05 | 0.0 |
| 11 | v3_lr1e5_r4_a2x_attn | v3 | 0.8378 | -0.0023 | 0.4549 | 0.4540 | 0.5220 | 1e-05 | 4 | 8 | 5.0 | 0.05 | 0.0 |
| 12 | v3_lr5e6_r16_attn | v3 | 0.8378 | -0.0023 | 0.4534 | 0.4534 | 0.5217 | 5e-06 | 16 | 32 | 5.0 | 0.05 | 0.0 |
| 13 | v3_lr1e5_r4_a1x | v3 | 0.8362 | -0.0038 | 0.4543 | 0.4543 | 0.5220 | 1e-05 | 4 | 4 | 5.0 | 0.05 | 0.0 |

## Key Findings

- **Best adapter**: `v3_lr5e6_r16_all` (EM = 0.8506, delta = +0.0106)
- **Worst adapter**: `v3_lr1e5_r4_a1x` (EM = 0.8362, delta = -0.0038)
- **5/13** adapters improved over base

---

*Generated: 2026-03-13 16:40*
*Full JSON: `/common/users/sl2148/Public/yang_ouyang/projects/LostInTheSecond/artifacts_local/lora_sweep_wr/_gsm8k_vllm_eval/all_results.json`*
