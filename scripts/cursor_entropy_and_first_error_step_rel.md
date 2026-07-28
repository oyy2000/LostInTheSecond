# Entropy and first error step relationship
_Exported on 4/27/2026 at 11:21:07 EDT from Cursor (3.2.11)_

---

**User**

这里有一些sample@results/gsm8k_3b_multi_sample，用Qwen 2.5 3B model生成的，我需要你帮我验证entropy和首个错误step的关系，现在应该只需要这一步以后的步骤了： Step 4. Compute step-level uncertainty. 你可以用GPU 0- 1, Step 1. Generate reasoning trajectories.

- Model: Qwen2.5-3B-Instruct

- Dataset: GSM8K, MATH500

- Sampling: temperature=0.7, top_p=0.95

- Keep incorrect trajectories only.

Step 2. Segment trajectories into reasoning steps.

- Split by newline, "Step", equation blocks, or sentence-level fallback.

- Remove final answer line.

Step 3. Label first-error step.

- Use GPT-4/Claude as annotator.

- Manually inspect 50 examples.

- Store τ for each trajectory.

Step 4. Compute step-level uncertainty.

- mean entropy

- max entropy

- entropy delta

- mean logprob

- min logprob

- logprob drop

Step 5. Evaluate localization.

- Exact match

- within ±1

- within ±2

- Spearman correlation between uncertainty score and error proximity.


---

**Cursor**

I'll start by exploring the project structure and the sample results to understand what we're working with.

Let me also check the existing scripts and source code for any related work.

I'll wait for both exploration agents to return before proceeding with the implementation plan.

Now I have a clear picture. The data has:
- `raw_cot_n8.jsonl`: trajectories with steps already segmented
- `first_error/gpt_first_error_cache.jsonl`: GPT-annotated first error step (`tau`) for incorrect trajectories

Now I need to compute step-level entropy/logprob using the model. Let me check existing src code and what's available.

Now let me check the existing delimiter entropy collection script and the sweep engine to understand the full pipeline.

Now let me check if the logprobs have already been collected, and look at the existing analysis script.

The delimiter_entropy directory is empty — logprobs haven't been collected yet. Let me also check the keystep_utils and understand the full data flow.

Now I have a complete picture. Let me plan and implement this. The pipeline needs:

1. **Step 4**: Collect token logprobs via vLLM (GPUs 0-1), then compute step-level uncertainty metrics
2. **Step 5**: Evaluate localization (exact match, within +/-1, +/-2, Spearman correlation)

The existing `10_0_collect_delimiter_entropy.py` already handles logprob collection via vLLM. I'll create a unified script that:
1. Collects logprobs (reusing the existing pattern)
2. Computes the 6 step-level uncertainty metrics
3. Predicts the error step using each metric
4. Evaluates localization

Let me write the scripts.

Now let me create the Step 4b script that computes the 6 uncertainty metrics and the Step 5 evaluation script.

Now let me create the visualization script for Step 5.

Now let me run the logprob collection on GPUs 0 and 1.

Shard 0 failed. Let me check the error log.

Found it: `vllmdebug` has vllm 0.16.0. Let me wait for the rest and also check `rl_steer_yang`.

Good. `vllmdebug` has vllm 0.16.0 and `rl_steer_yang` has 0.8.5. Let me check which one was used for the existing scripts and which has the right API.

Good, vllm imports fine in `vllmdebug`. Now I need to update the script to use the correct Python interpreter. Let me modify the script to use the vllmdebug env's Python for the shard subprocesses.

Also need to make sure the subprocess uses the same Python:

Now let me also check if the existing `10_0` script was run with a specific env, and check if `yaml` is available in vllmdebug:

Good. Now let me clean up the failed shard output and re-run:

The script is running but no output yet (the shards are starting up). Let me wait for it — this will take a few minutes for model loading + inference on 1666 trajectories across 2 GPUs.

No output at all after 2+ minutes. The parent process might be waiting for the shards. Let me check if the shard processes are running and if there are any shard logs:

The shards have already completed (GPU memory is freed, shard files are ~13MB each). The parent process should be finishing up. Let me check:

All 1666 records processed. The logs were cleaned up (success). Let me check if the parent finished:

Logprob collection complete: 1666 records in 152 seconds. Now let me run Step 4b + Step 5 (compute metrics and evaluate localization).

Excellent results. Now let me generate the figures:

Let me fix the deprecation warning and check the figures:

Let me view the key figures:

All done. Here's the summary:

**Step 4 (logprob collection)**: Collected per-token logprobs for all 1666 incorrect GSM8K trajectories (with valid tau) using Qwen2.5-3B-Instruct via vLLM on GPUs 0-1. Took ~152s.

**Step 4 (uncertainty metrics)**: Computed 6 step-level metrics for each trajectory:

| Metric | Exact Match | Within +/-1 | Within +/-2 | Spearman rho | MAE |
|---|---|---|---|---|---|
| mean_entropy | 0.124 | 0.348 | 0.532 | 0.108 | 2.76 |
| max_entropy | 0.214 | 0.480 | 0.665 | 0.144 | 2.17 |
| entropy_delta | 0.242 | 0.459 | 0.653 | 0.240 | 2.06 |
| mean_logprob | 0.148 | 0.383 | 0.577 | 0.139 | 2.59 |
| min_logprob | **0.261** | **0.508** | **0.685** | 0.206 | **2.05** |
| logprob_drop | 0.238 | 0.468 | 0.661 | **0.227** | 2.05 |

Key findings:
- `min_logprob` is the best single-metric localizer: 26.1% exact match, 50.8% within +/-1, 68.5% within +/-2.
- `entropy_delta` has the highest Spearman correlation (0.24) with the gold error step.
- The entropy profile plot shows a clear spike at the error step (tau=0), confirming that the model's uncertainty does increase at the first error.
- The box plots show statistically significant differences (p < 0.001) between error and non-error steps for all metrics.

Scripts created:
- `scripts/4_0_collect_step_logprobs.py` — multi-GPU logprob collection
- `scripts/4_1_compute_uncertainty_and_evaluate.py` — metrics + localization evaluation
- `scripts/5_0_localization_figures.py` — visualization

Results saved to `results/gsm8k_3b_multi_sample/step_uncertainty/`, figures to `figures/step_uncertainty/`.

---

**User**

计算当前step的entropy异动下一步为错误步骤的概率，或者上一步为错误步骤的概率

---

**Cursor**

Let me first understand the current data structure and what we're working with.

Good. Now I'll write the analysis script. The core idea: for every step, given its entropy metrics, compute P(next step is error) and P(previous step is error).

Hmm, empty output. Let me check:

The file is empty. The heredoc didn't work properly. Let me write it differently:

The file doesn't exist. The heredoc with single quotes around PYEOF should have worked. Let me try a different approach:

The old_string might not be unique. Let me read the file to see current state:

Good, the file has 92 lines so far. Let me append the remaining functions:

There must be a trailing newline issue. Let me check:

Something is wrong with the shell. Let me try a fresh approach and write the complete file at once using a Python script:

The shell seems stuck. Let me open a fresh one:

Something is very wrong with the shell. Let me try a completely new shell:

The shell is returning empty for everything. Let me try the Write tool directly to create the full script. I'll write it in parts.

I see the issue - the `# STUB2` replacement is failing. Let me try with exact match:

The StrReplace tool keeps failing. Let me just delete and rewrite the whole file using Write:

I'll write the script using a Python helper to avoid heredoc/encoding issues:

Good. Now let me run it:

The shell is still returning empty. Let me try a completely different approach:

The shell tool seems to be in a broken state. Let me use a subagent to run the script:

The shell tool is broken in this session. Let me try a background command approach which writes to a file:

Results are in. Let me view the generated figures:

分析完成。结果如下：

**核心发现：当前 step 的 entropy 异动对相邻 error step 的预测力有限但有方向性信号。**

**P(next step = error | current step features)**
- Logistic regression AUC = 0.5435 (6 metrics联合)
- 最强单变量信号：Min LogProb (AUC=0.5174)、Entropy Delta (AUC=0.5120)
- 经验概率：Mean Entropy 最高 decile 的 P(next=error)=16.2%，最低 decile 为 11.7%，约 1.39x

**P(prev step = error | current step features)**
- Logistic regression AUC = 0.5892 (明显优于 next 方向)
- 最强单变量信号：LogProb Drop (AUC=0.5725)、Entropy Delta (AUC=0.5596)
- LogProb Drop 最高 decile 的 P(prev=error)=15.6%，最低 decile 为 8.9%，1.74x
- 这说明错误步骤的"余波"比"前兆"更容易被 entropy 捕捉到

**Logistic regression 系数解读**
- 预测 next=error：Entropy Delta 负系数最大 (-0.34)，说明 entropy 突然下降后下一步更可能出错
- 预测 prev=error：LogProb Drop 正系数最大 (+0.42)，说明当前步 logprob 突然下降意味着上一步可能已经出错（error cascade 效应）

总体来看，单步 entropy 特征对相邻 error 的预测力偏弱（AUC 在 0.54-0.59），但 "prev=error" 方向的信号更强，符合直觉：错误发生后，后续步骤的 logprob 会出现明显 drop。

所有结果保存在 `results/gsm8k_3b_multi_sample/step_uncertainty/conditional_error_prob_summary.json`，图表在 `figures/step_uncertainty/`。

---

**User**

做以下实验： AUC 0.54-0.59 在 ML 里通常被认为是"勉强有信号"，但放在你这个 setting 里要重新评估，原因有三：

错误的 base rate 大概是 12%（你给的 deciles 11.7-16.2%）。在这种 imbalanced 的设定下，AUC 是个 lousy metric，precision-recall 曲线和 lift 才是。Mean Entropy top decile 的 lift 是 1.39x，LogProb Drop top decile 是 1.74x —— 1.74x 在 imbalanced detection 里其实算可用的弱信号。
你的 trigger 不需要每一步都判对。每个轨迹只要在它的 ~10-20 个 step 里有一个能被 trigger 命中，就够触发一次修复。所以应该看的不是"每步 AUC"，而是 trajectory-level：能在 error step 之前 / 之上触发的概率。AUC 0.59 在每步上意味着每条轨迹的 trigger 能命中错误的概率会显著高于 0.59。
"prev=error 信号比 next=error 强"是一个非常重要的发现，不是 noise。它对应的物理含义是：模型在出错之后，下一步开始"勉强续接"，logprob 会掉。这意味着 trigger 应该是 lookback 型（"我现在 logprob 掉了 → 上一步可能错了，回去 repair 上一步"），而不是 lookahead 型（"我现在 entropy 高 → 这一步可能错"）。

这个 asymmetry 直接改变了你方法的 architecture，不只是阈值。我下面的实验设计都是围绕这个写的。

实验设计
按重要性排序，下面五个都建议做。
实验 1（必做，主结果图）：iso-compute Pareto 曲线
这是审稿人会看的第一张图，也是你 motivation 的最终归宿。

X 轴：total generation tokens（不是 sample 数，因为不同方法每个 sample 长度不一样）
Y 轴：accuracy
模型 × 数据集：(llama-3.2-3B, Qwen2.5-3B) × (GSM8K, MATH500)，4 个子图
每个子图上画几条曲线（每条扫超参产生 Pareto 前沿）：

Greedy（单点 anchor）
Self-consistency，N ∈ {1, 2, 4, 8, 16, 32}
Random-repair（在每步以概率 p 触发 repair，扫 p）—— 这是关键 baseline，证明"任何修复都能赢 SC"是不成立的
Entropy-triggered lookback repair（你的方法），扫阈值
可选：Self-Eval Guided (Xie 2023) at matched compute
可选上限：小 PRM-based（说明你跟 PRM 的 gap，但不是必须 beat）



胜利条件：你的曲线在合理 compute 区间 Pareto 主导 SC 和 random-repair。即使 AUC 只有 0.59，只要能 dominate，就够发文章。
实验 2（必做）：定义"trajectory-level trigger 命中率"
把 step-level AUC 0.59 翻译成 trajectory-level 指标，避免审稿人盯着 0.59 不放。具体定义两个量：

Hit@k：trigger 在错误 step 附近 ±k 步以内触发的轨迹比例。k=0, 1, 2 都画。
False-trigger rate：在 correct 轨迹上的平均触发次数（每条轨迹的 cost 直接跟这个挂钩）。

然后画 ROC-style：x 轴 false-trigger rate per trajectory，y 轴 Hit@1。这条曲线决定了你阈值能选到什么操作点。我猜你会看到 Hit@1 在 false-trigger=1.0/traj 时能到 0.6-0.7，这就足够支撑实验 1 的结果了。
实验 3（高优先级）：lookback vs lookahead vs symmetric trigger
直接做 ablation 验证你这个 asymmetry 发现：

变体 A: lookahead trigger —— "current step entropy 高 → repair current"
变体 B: lookback trigger —— "current step logprob drop 大 → repair previous"  ← 这是你应该 push 的版本
变体 C: symmetric —— current 触发，repair 当前 ± 1 都试

如果 B 显著优于 A，这就是一个干净的 contribution，写 paper 时是一个独立的 finding。
实验 4（高优先级）：组合信号 vs 单一 entropy
你 logistic regression 的 AUC 0.59 是 6 个 metric 联合的，单变量最高 0.57。这里有两个低成本提升点：

加一个 self-eval 信号：在 step 边界让模型自己续写一个 token "Correct/Wrong" 的概率（Xie 2023 的 PMI 思路，但只在 trigger 决策时用，不做 beam search）。这只多 1-2 次 forward，cost 很小，但应该能把 AUC 推到 0.65-0.7。重要的是这不需要训 PRM。 Self-Evaluation Guided Beam Search (Xie et al., NeurIPS 2023)
这个有正式论文。做法是 step-level beam search，每生成一个 step 后让模型自己评分。评分用一个 PMI-style 公式：
score(step) = log p(step | context) + λ · log p("correct" | step, context, eval prompt)
第一项是生成概率（log-prob 信号），第二项是用 few-shot prompt 让模型 self-evaluate "this step is correct"。这等价于 用同一个 model 当 verifier，不引入新参数。然后按 score 做 beam pruning。
放到你的方法里：可以在每个 step 结束后算这个 score，低于阈值就 resample 这一步（不 rollback、不引外部 PRM），完全契合你 figure 1 显示的 "repair > rollback" 信号。 
学一个 step-level head：把 entropy / logprob / position embedding 喂给一个 2 层 MLP，用你已经标注的 first-error 数据训。这等于一个 micro-PRM 但只用模型自己的 hidden state，参数量极小，inference cost 可以忽略。

实验里报：单一 entropy → logistic combo → +self-eval → +small head，看 AUC 怎么爬升，然后用每个版本跑实验 1。
实验 5（可选，但很 sell）：错误位置分布的描述性统计
补一张图：first-error step 的相对位置分布（τ/N 直方图），分模型 × 分数据集。这张图本身就很有用：

如果 GSM8K 的 first-error 集中在后半段、MATH500 散布更均匀，就解释了为什么不同 dataset 上你的 trigger 表现不一样。
这张图也是你 motivation 的视觉锚点。


关于"我用纯 entropy 而不是 Self-Eval Guided"
我建议把它们当作两个互补的信号，而不是二选一。理由：

纯 entropy / logprob：0 额外 forward，cost 真的可以忽略不计。
Self-Eval (PMI)：每个 step 边界 1-2 次额外 forward，cost 很小但非零，AUC 通常会高 0.05-0.10。

Paper 里你可以这样组织：

主线方法用 entropy-only（强调 zero-overhead）。
一个 "+ self-eval" 变体作为更强但稍贵的版本。
实验 1 的 Pareto 图里两条线都画，让 reader 自己看 trade-off。

这样你既有"最 cheap"的 selling point，又能展示方法的 scalability，审稿人很难说你方法 cheap 是因为弱。

---

**Cursor**

This is a substantial experimental plan. Let me first understand the existing codebase and data, then implement systematically.

Let me first explore the existing codebase to understand what infrastructure is already in place — sweep engine, repair engine, existing results, and available data for both models and datasets.

While those explore, let me start on Exp 5 (quick descriptive stats) and read the key files I need:

Now let me read the prompt_templates and check existing sweep results:

Now let me check what existing sweep results we have and GPU availability:

Excellent. Now I have a comprehensive picture. Let me also check the Qwen sweep results and the first_error data for both models:

Now I have a complete picture. Key findings:

- GPUs 0, 1, 6, 7 are free (40GB A100s)
- No Qwen sweep results exist yet (only Llama sweeps)
- GSM8K Qwen first_error: 1817 records, MATH500 Qwen first_error: 6820 records
- Llama GSM8K first_error: only 133 records (need more)
- Existing scripts `5_1_tau_distribution.py`, `5_2_trajectory_trigger_hitrate.py`, `6_0_repair_sweep.py` already exist

Let me check what those existing scripts look like:

The `6_0_repair_sweep.py` is incomplete (has placeholders). This is the main script I need to build out. Let me also check the `step_scorers.py`:

Now I have a thorough understanding of the codebase. The `6_0_repair_sweep.py` is incomplete. I need to build the complete repair sweep pipeline. Let me plan the approach and start implementing.

The key insight from the user's analysis: the architecture should be **lookback-triggered** (detect logprob drop at step t, repair step t-1). I need to build:

1. **Exp 5** (quick): tau/N distribution -- script exists, just run it
2. **Exp 2** (quick): trajectory-level trigger hit rate -- script exists, just run it  
3. **Exp 3 + Exp 1** (main): Complete the repair sweep with lookback/lookahead/symmetric/random triggers, then plot Pareto curves
4. **Exp 4**: Signal ladder (later, after main results)

Let me start by running the quick scripts (Exp 5 and Exp 2), then build the full repair sweep.

Now let me build the main repair sweep script. The existing `6_0_repair_sweep.py` has the right structure but is incomplete. I need to complete it with all phases. Let me write the complete version:

Exp 5 done. Exp 2 needs a sys.path fix. Let me fix and rerun:

Now let me build the main repair sweep script. This is the most complex piece — it needs to:
1. Generate greedy drafts
2. Compute per-step logprobs
3. Apply triggers (lookback/lookahead/symmetric/random) at various thresholds
4. Generate repair suffixes
5. Evaluate with SC baselines
6. Plot Pareto curves

