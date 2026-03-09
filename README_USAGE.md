# LostInTheSecond 脚本触发手册

这个文档是“可直接复制运行”的触发说明，覆盖当前目录下主要脚本的常见用法。

## 0. 运行前准备

- 工作目录：`LostInTheSecond/`
- 建议先安装依赖：

```bash
pip install -r ../fact-enhancement/requirements.txt
pip install -r ./lm-evaluation-harness/requirements.txt
```

- 如果要运行 GPT 改写脚本，准备 `.env`：

```bash
OPENAI_API_KEY=...
```

---

## 1) `00_PRM_boards.py`

用途：
- 对 baseline 生成做 PRM 打分
- 产出 step2 wrong 子集 ID
- 导出负样本集（后续给 GPT 修复）

### 基础触发

```bash
python 00_PRM_boards.py \
  --runs-root ./runs/baseline_qwen25_3b_math500 \
  --score-model Qwen/Qwen2.5-Math-PRM-7B \
  --out-prm-json ./artifacts/prm_scores_baseline.json \
  --out-ids ./artifacts/s_step2_wrong_ids.json \
  --out-neg ./artifacts/negative_samples_step2_wrong.json
```

### 仅复用已有 PRM 结果画图

```bash
python 00_PRM_boards.py --only-plot --reuse-prm-json
```

### 常用参数

- `--baseline-samples`：直接指定 `samples_*.jsonl`
- `--step-split-mode`：`double_newline|single_newline|auto|regex`
- `--max-subset`：step2 wrong 子集大小
- `--plot-threshold`：画 per-step correctness 曲线阈值

---

## 2) `01_construct_datasets_by_GPT.py`

用途：
- 用 GPT 对 step2 做修复
- 构建两份数据：
  - `ds2_fix_step2_gpt.json`
  - `ds2_wait_recompute_gpt.json`
- 输出审计文件 `gpt_fix_step2_audit.json`

### 基础触发

```bash
python 01_construct_datasets_by_GPT.py \
  --env-file .env \
  --in-file ./artifacts/samples_math500_ds2_wait_recompute.json \
  --out-ds2-correct ./artifacts/samples_math500_ds2_fix_step2_gpt.json \
  --out-ds2-wait ./artifacts/samples_math500_ds2_wait_recompute_gpt.json \
  --audit-json ./artifacts/gpt_fix_step2_audit.json \
  --model gpt-5.1 \
  --judge-first \
  --limit 0
```

### 强制全部改写（忽略 judge）

```bash
python 01_construct_datasets_by_GPT.py --force-rewrite --limit 0
```

```
python 01_construct_gsm8k_step2_by_GPT.py --in-file ./artifacts/vectors_16_ds2_fix_step2_incorrect_only_id300+_step2_marker/Qwen_Qwen2.5-3B-Instruct_applied/gsm8k_openai_train_step2fix_fix_step2/Qwen2.5-3B-Instruct_L1_BASELINE/samples_gsm8k_openai_train_incorrect_recomputed.jsonl --out-file ./artifacts/vectors_16_ds2_fix_step2_incorrect_only_id300+_step2_marker/Qwen_Qwen2.5-3B-Instruct_applied/gsm8k_openai_train_step2fix_fix_step2/Qwen2.5-3B-Instruct_L1_BASELINE/samples_gsm8k_openai_train_corrected_second.jsonl  --model gpt-5.1 --judge-first --limit 2
```

### 常用参数

- `--judge-first`：先判断 step2 对错，再决定是否改写
- `--force-rewrite`：无论 judge 结果都改写
- `--temperature` / `--max-output-tokens`
- `--retries` / `--sleep-base`

---

## 3) `02_extract_vectors_large.py`

用途：
- 从 pos/neg 成对样本抽取 steering vector
- 当前已支持 **Step2 marker 提取版本**：
  - 在 pos/neg 中插入 `<<STEP2_END>>`
  - 通过裁剪到 marker 并 `read_token_index=-1` 对齐到 marker 位置

> 该脚本是“配置驱动”，通过文件顶部常量切换模式。

### 触发方式

```bash
python 02_extract_vectors_large.py
```

### 运行前要改的核心常量（文件顶部）

- `EXPERIMENT_MODE`：`GPT_STEP2_FIX | GPT_REWRITE | LARGE_MODEL`
- `TARGET_MODEL`
- `EXTRACTION_VARIANT`：`step2_marker | baseline`
- `DATASET_VARIANT`（在 `GPT_STEP2_FIX` 分支）
- `MIN_DOC_ID`（当前逻辑会筛 `id > MIN_DOC_ID`）

### 输出位置（示例）

- `./artifacts/vectors_{N}_ds2_fix_step2_incorrect_only_id300+_step2_marker/.../steering_vector.pt`

---

## 4) `03_exp_multi_large_pair.py`

用途：
- 批量跑 `lm_eval` + `steer_hf`
- 多 GPU 调度、断点续跑（`runs.json`）
- 扫 layer × lambda 网格

> 该脚本也是“配置驱动”，通过文件顶部常量控制。

### 触发方式

```bash
python 03_exp_multi_large_pair.py
```

### 关键配置项（文件顶部）

- `EXPERIMENT_MODE`：`GPT_REWRITE | LARGE_MODEL | GPT_STEP2_FIX`
- `MODEL_TO_LAYERS`：要测试的层
- `STEER_LAMBDAS`：系数网格
- `LIMIT` / `BATCH_SIZE` / `TASKS`

### 新增 apply 窗口配置（已接入 `steer_hf`）

- `STEER_APPLY_MODE`：
  - `prefix`：仅前缀窗口（默认）
  - `step2_window`：Step2 周围窗口
  - `all`：全程施加
- `STEER_MIN_TOKEN` / `STEER_MAX_TOKEN`：前缀窗口边界（默认前 128）
- `STEER_WINDOW_CENTER` / `STEER_WINDOW_PRE` / `STEER_WINDOW_POST`：Step2 窗口

---

## 5) `03_resample_wrong_subset.py`

用途：
- 对 step2 wrong 子集做随机重采样（温度采样）

### 基础触发

```bash
python 03_resample_wrong_subset.py \
  --wrong-ids ./artifacts/s_step2_wrong_ids.json \
  --runs-root ./runs/baseline_qwen25_3b_math500 \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --num-resamples 50 \
  --temperature 1.0 \
  --top-p 0.7 \
  --output ./artifacts/wrong_subset_resamples_t1p0_top0p7.jsonl
```

### 常用参数

- `--max-new-tokens`
- `--batch-return`（单次 generate 返回条数）
- `--dtype`：`float16|bfloat16|float32`

---

## 6) `04_multi_card_PRM.py`

用途：
- 对 steering 实验输出做多 GPU PRM 打分
- 合并结果并自动出图

### 基础触发

```bash
python 04_multi_card_PRM.py \
  --results-root ./artifacts/vectors_16_ds2_fix_step2_incorrect_only/Qwen_Qwen2.5-3B-Instruct_applied/hendrycks_math_500_step2fix_fix_step2 \
  --num-gpus 8
```

### 只画图（不重新打分）

```bash
python 04_multi_card_PRM.py --only-plot
```

### 和 baseline 做 step-correctness 对比

```bash
python 04_multi_card_PRM.py --compare-to-baseline
```

---

## 7) `05_finetune_lora_same_dataset.py`

用途：
- 使用现有 pair 数据（默认 `pos_response`）做 LoRA SFT

### 基础触发

```bash
python 05_finetune_lora_same_dataset.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --dataset-path ./artifacts/samples_math500_ds2_fix_step2_gpt.json \
  --output-dir ./artifacts/lora_qwen25_3b_ds2_fix_step2 \
  --num-train-epochs 2 \
  --learning-rate 2e-4


python 05_finetune_lora_same_dataset.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --dataset-path ./artifacts/samples_math500_ds2_wait_recompute_gpt.json \
  --output-dir ./artifacts/lora_qwen25_3b_ds2_fix_step2 \
  --num-train-epochs 2 \
  --learning-rate 2e-4
```

### 常用调参

- LoRA：`--lora-r --lora-alpha --lora-dropout --target-modules`
- 训练：`--per-device-train-batch-size --gradient-accumulation-steps`
- 精度：`--bf16/--no-bf16`、`--fp16`
- 截断：`--max-length`

---

## 8) `run_preliminary_8gpu.py`（一键 baseline + extract + eval）

用途：
- 统一跑 baseline
- 提取两套向量
- 8 卡并行 steering eval

### 基础触发

```bash
python run_preliminary_8gpu.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --task hendrycks_math_500 \
  --layers auto \
  --lambdas 0.25,0.5,1.0,2.0,-0.5,-1.0 \
  --gpus 0,1,2,3,4,5,6,7
```

### 跳过 baseline 或 extraction

```bash
python run_preliminary_8gpu.py --skip-baseline
python run_preliminary_8gpu.py --skip-extract
```

### 常用参数

- `--dataset-a --dataset-b`
- `--read-token-index`（marker 方案常配合使用）
- `--gen-kwargs` / `--batch-size` / `--limit`

---

## 9) `06_eval_lora_effect.py`（对比 LoRA 前后效果）

用途：
- 在同一个 task 上分别评测 Base 与 LoRA（同一模型、同一生成参数）
- 自动解析 `exact_match` 并输出对比汇总

### 基础触发

```bash
python 06_eval_lora_effect.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --lora-path ./artifacts/lora_qwen25_3b_ds2_fix_step2/final_adapter \
  --task hendrycks_math_500 \
  --harness-dir ./lm-evaluation-harness \
  --output-root ./runs/lora_eval_compare \
  --batch-size 16 \
  --dtype float16 \
  --cuda-visible-devices 0
```

### 先做小样本 smoke test

```bash
python 06_eval_lora_effect.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --lora-path ./artifacts/lora_qwen25_3b_ds2_fix_step2/final_adapter \
  --task hendrycks_math_500 \
  --limit 50 \
  --batch-size 8 \
  --cuda-visible-devices 0
```

### 只跑单边（调试）

```bash
python 06_eval_lora_effect.py --skip-lora   # 只跑 base
python 06_eval_lora_effect.py --skip-base   # 只跑 lora
```

### 常用参数

- `--gen-kwargs`：默认 `max_gen_toks=2048,temperature=0,do_sample=False`
- `--limit`：评测样本数（0 表示全量）
- `--batch-size`：评测 batch 大小
- `--dtype`：`float16|bfloat16|float32`
- `--cuda-visible-devices`：指定显卡

### 输出说明

默认输出目录：`./runs/lora_eval_compare/`

- `base/.../results_*.json`
- `lora/.../results_*.json`
- `comparison_summary.json`

`comparison_summary.json` 关键字段：

- `base.score_exact_match`
- `lora.score_exact_match`
- `delta_lora_minus_base`（LoRA - Base）

### 常见问题

- `ModuleNotFoundError: No module named 'sacrebleu'`

```bash
pip install sacrebleu
# 或一次性装全
pip install -r ./lm-evaluation-harness/requirements.txt
```

- `LoRA adapter not found`：检查 `--lora-path` 是否指向 `final_adapter`

---

## 10) `05_prefill_pos_tail.py`（基于前两步 prefill 重生成后续）

用途：
- 从样本中取前缀（优先：`step1 + 修复后的step2`）
- 用指定模型续写后续 tail
- 产出新的 `pos_response_prefill`（可选覆盖 `pos_response`）

### 基础触发

```bash
python 05_prefill_pos_tail.py \
  --in-file ./artifacts/samples_math500_ds2_fix_step2_gpt.json \
  --step2-source-file ./artifacts/samples_math500_ds2_fix_step2_gpt.json \
  --out-file ./artifacts/samples_math500_ds2_fix_step2_gpt_prefill.json \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --keep-steps 2 \
  --
  --max-new-tokens 768 \
  --dtype bfloat16
```

### 先小样本试跑

```bash
python 05_prefill_pos_tail.py \
  --limit 20 \
  --in-file ./artifacts/samples_math500_ds2_fix_step2_gpt.json \
  --out-file ./artifacts/samples_math500_ds2_fix_step2_gpt_prefill_smoke.json
```

### 常用参数

- `--keep-steps`：默认 2（建议保持 2）
- `--max-new-tokens`：续写长度上限
- `--do-sample --temperature --top-p`：是否采样续写
- `--no-overwrite-pos-response`：不覆盖原 `pos_response`
- `--require-exact-match/--no-require-exact-match`：是否仅处理正确样本

### 输出字段（样本级）

- `pos_response_original`
- `pos_response_prefill`
- `pos_response`（默认会被覆盖为 prefill 结果）
- `pos_steps`（按新结果重切分）

---

## 11) `08_extract_vector_first2steps.py`（只抽前两步向量）

用途：
- 只使用前两步（`pos[:2]` vs `neg[:2]`）构建 steering pairs
- 用 `train_steering_vector` 提取向量，避免后续步骤噪声

### 基础触发

```bash
python 08_extract_vector_first2steps.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --dataset-path ./artifacts/samples_math500_ds2_fix_step2_gpt.json \
  --out-dir ./artifacts/vectors_first2steps \
  --layers all \
  --read-token-index -1
```
samples_math500_ds2_fix_step2_gpt_prefill.json
/common/users/sl2148/Public/yang_ouyang/projects/LostInTheSecond/artifacts/samples_math500_ds2_fix_step2_gpt.json

### 常用参数

- `--layers`：`all` 或逗号分隔层号（如 `6,16,17`）
- `--max-samples`：限制训练对数（0 表示全量）
- `--min-doc-id`：只用 `doc.id > min_doc_id`
- `--dtype`：加载模型精度

### 输出位置（示例）

- `./artifacts/vectors_first2steps/Qwen_Qwen2.5-3B-Instruct_first2steps/steering_vector.pt`
- `./artifacts/vectors_first2steps/Qwen_Qwen2.5-3B-Instruct_first2steps/extract_meta.json`

---

## 12) `09_steer_vector_exp_token_ranges.py`（仅比较 prefix vs all）

用途：
- 固定比较两种施加范围：
  - `prefix`：只在前缀 token 窗口施加
  - `all`：从头到尾全程施加
- 对同一组 `model × layer × lambda` 同时跑两套结果

### 基础触发

```bash
python 09_steer_vector_exp_token_ranges.py \
  --vector-path ./artifacts/vectors_first2steps/Qwen_Qwen2.5-3B-Instruct_first2steps/steering_vector.pt \
  --models Qwen/Qwen2.5-3B-Instruct \
  --layers 6,16,17 \
  --lambdas -1.0,-0.5,0.0,0.5,1.0 \
  --tasks hendrycks_math_500 \
  --steer-min-token 0 \
  --steer-max-token 128 \
  --gpus 0,1,2,3,4,5,6,7
```

### 常用参数

- `--limit`：评测样本数（默认 400）
- `--batch-size` / `--gen-kwargs` / `--dtype`
- `--steer-min-token --steer-max-token`：prefix 窗口边界
- `--output-root`：输出根目录

### 输出结构

- `./runs/09_token_range_exp/prefix/...`
- `./runs/09_token_range_exp/all/...`
- `./runs/09_token_range_exp/plan.json`

---

## 13) 推荐执行顺序（当前工程）

1. `00_PRM_boards.py`：找 step2 wrong 子集
2. `01_construct_datasets_by_GPT.py`：构建 DS2 数据
3. `02_extract_vectors_large.py`：抽向量（可用 `step2_marker`）
4. `03_exp_multi_large_pair.py`：批量 apply 评测（可设前 128 / step2 窗口）
5. `04_multi_card_PRM.py`：多卡 PRM 分析出图
6. （可选）`05_finetune_lora_same_dataset.py`：LoRA 微调
7. （可选）`06_eval_lora_effect.py`：量化 LoRA 提升
8. （推荐新链路）`05_prefill_pos_tail.py`：先重生成修复后续写
9. （推荐新链路）`08_extract_vector_first2steps.py`：只抽前两步向量
10. （推荐新链路）`09_steer_vector_exp_token_ranges.py`：比较 prefix 与 all

---

## 14) 常见问题

- `steering_vectors` import 报错：
  - 安装 `steering-vectors` 包，并确认当前环境与运行环境一致。
- 找不到 `samples_*.jsonl`：
  - 检查 `--runs-root` 是否正确，或显式传 `--baseline-samples`。
- 多卡没跑满：
  - 检查 `nvidia-smi`、`NUM_GPUS/GPUS`、以及每卡显存门限配置。
