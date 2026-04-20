# LostInTheSecond 脚本手册

## 目录结构

```
scripts/
├── data_prep/           # 数据准备与诊断
│   ├── 00_PRM_boards.py                  # PRM 打分，找 step2-wrong 子集
│   ├── 00_check_gsm8k_correctness.py     # 重算 GSM8K 正确性，导出 incorrect 样本
│   ├── 01_construct_datasets_by_GPT.py   # GPT 修复 step2 (MATH-500)
│   ├── 01_construct_gsm8k_step2_by_GPT.py # GPT 修复 step2 (GSM8K)
│   ├── 03_resample_wrong_subset.py       # step2-wrong 子集温度重采样
│   └── 05_prefill_pos_tail.py            # 取前两步 prefix + 模型续写 tail
│
├── vector/              # 向量提取
│   ├── 02_extract_vectors_large.py       # pos/neg pairs → steering vector
│   └── 08_extract_vector_first2steps.py  # 仅前两步提取向量
│
├── steer_eval/          # Steering 施加与评测
│   ├── 03_exp_multi_baseline.py          # 多 GPU baseline 评测
│   ├── 03_exp_multi_large_pair.py        # 多 GPU steer_hf, layer×lambda 网格
│   └── 09_steer_vector_exp_token_ranges.py # prefix vs all 施加范围对比
│
├── lora/                # LoRA 微调
│   ├── 05_finetune_lora_same_dataset.py  # 单次 LoRA SFT
│   ├── 07_lora_sweep.py                  # Phase 1 超参搜索 (25 实验)
│   └── 11_sweep_v2_and_eval.py           # Phase 2 超参搜索 (16 实验)
│
├── eval/                # 评测与分析
│   ├── 04_multi_card_PRM.py              # 多 GPU PRM 打分 + 出图
│   ├── 06_eval_lora_effect.py            # LoRA vs Base 对比评测
│   ├── 07_prefill_depth_curve.py         # Prefill depth 诊断曲线
│   ├── 10_batch_eval_all_models.py       # 批量 GSM8K 评测 (HF backend)
│   └── 12_vllm_gsm8k_eval.py            # 批量 GSM8K 评测 (vLLM, 4GPU 并行)
│
├── slurm/               # SLURM 提交脚本
│   ├── gsm8k_vllm_eval.sh               # 提交 vLLM GSM8K 评测
│   └── lora_sweep_v3.sh                  # 提交 LoRA 训练 + 评测
│
├── utils.py             # 共享工具 (Qwen chat template 等)
├── prm_shared.py        # PRM 共享函数 (打分、step 切分、画图)
├── run_preliminary_8gpu.py  # 一键 baseline → 向量提取 → 评测
└── step_correctness_Debug.ipynb
```

## 存储布局

```
/home/youyang7/projects/LostInTheSecond/
├── scripts/              # 代码 (本地)
├── documents/            # 报告文档 (本地)
├── lm-evaluation-harness/ # lm-eval (本地, dev install)
└── artifacts/            # → symlink → /mnt/beegfs/.../artifacts (大文件)
    ├── lora_sweep/       # 所有 LoRA adapters + 评测结果 (~17G)
    ├── samples_*.json    # 训练数据
    └── ...

/mnt/beegfs/youyang7/projects/LostInSecond/
├── artifacts/            # 实际存储位置
└── logs/                 # SLURM 日志
```

## SLURM 提交

### 评测所有 LoRA 模型 (vLLM, 4 GPU 并行)

```bash
cd /home/youyang7/projects/LostInTheSecond
sbatch scripts/slurm/gsm8k_vllm_eval.sh

# 监控
squeue -u $USER
tail -f /mnt/beegfs/youyang7/projects/LostInSecond/logs/gsm8k_eval_*.out
```

### LoRA 训练 + 评测

```bash
sbatch scripts/slurm/lora_sweep_v3.sh
```

## 常用触发命令

### 1) 数据准备

```bash
# PRM 打分 (MATH-500)
python scripts/data_prep/00_PRM_boards.py \
  --runs-root ./runs/baseline_qwen25_3b_math500

# GPT 修复 step2 (GSM8K)
python scripts/data_prep/01_construct_gsm8k_step2_by_GPT.py \
  --in-file ./artifacts/... --model gpt-5.1 --judge-first --limit 0

# Prefill 续写
python scripts/data_prep/05_prefill_pos_tail.py \
  --in-file ./artifacts/samples_gsm8k_train_ds2_fix_step2_gpt.json \
  --model-id Qwen/Qwen2.5-3B-Instruct
```

### 2) 向量提取

```bash
# 全步向量
python scripts/vector/02_extract_vectors_large.py

# 仅前两步
python scripts/vector/08_extract_vector_first2steps.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --dataset-path ./artifacts/samples_math500_ds2_fix_step2_gpt.json
```

### 3) LoRA 训练

```bash
# 单次训练
python scripts/lora/05_finetune_lora_same_dataset.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --dataset-path ./artifacts/samples_gsm8k_train_ds2_fix_step2_gpt_prefill.json \
  --output-dir /mnt/beegfs/youyang7/projects/LostInSecond/artifacts/lora_new \
  --learning-rate 1e-4 --num-train-epochs 5

# Phase 1 sweep (25 实验)
python scripts/lora/07_lora_sweep.py --gpus 0,1,2,3

# Phase 2 sweep (16 实验)
python scripts/lora/11_sweep_v2_and_eval.py --gpus 0,1,2,3
```

### 4) 评测

```bash
# vLLM 并行评测 (推荐)
python scripts/eval/12_vllm_gsm8k_eval.py --gpus 0,1,2,3

# 仅重新生成报告
python scripts/eval/12_vllm_gsm8k_eval.py --summary-only

# LoRA vs Base 单模型对比
python scripts/eval/06_eval_lora_effect.py \
  --lora-path ./artifacts/lora_sweep/lr_1e-4/final_adapter
```

## 推荐执行顺序

**链路 A — MATH-500 Steering Vector：**
`data_prep/00_PRM_boards` → `data_prep/01_construct_datasets_by_GPT` → `data_prep/05_prefill_pos_tail` → `vector/02_extract_vectors_large` → `steer_eval/03_exp_multi_large_pair` → `eval/04_multi_card_PRM`

**链路 B — GSM8K LoRA 微调：**
`data_prep/00_check_gsm8k_correctness` → `data_prep/01_construct_gsm8k_step2_by_GPT` → `data_prep/05_prefill_pos_tail` → `lora/05_finetune_lora_same_dataset` → `lora/07_lora_sweep` → `eval/12_vllm_gsm8k_eval`
