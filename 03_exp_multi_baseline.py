#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
import signal
import sys
import shlex
import json

# ==========================================
# 1. 核心控制开关
# ==========================================
# 选项: "GPT_REWRITE" (3B实验) 或 "LARGE_MODEL" (0.5B实验) 或 "GPT_STEP2_FIX"
EXPERIMENT_MODE = "BASELINE" 

# ==========================================
# 2. 公共配置 (两个实验共用)
# ==========================================
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]  # 使用的 GPU 列表
TASKS = "gsm8k_openai_train" # "gsm8k_cot_zeroshot_unified" 
GEN_KWARGS = "max_gen_toks=2048,temperature=0,do_sample=False"
MODEL = "steer_hf"
NUM_FEWSHOT = "0"
APPLY_CHAT_TEMPLATE = True
LIMIT = None #400
BATCH_SIZE = "32"
APPLIED_MODEL = "Qwen/Qwen2.5-3B-Instruct" # "meta-llama/Llama-3.2-1B-Instruct" #"meta-llama/Llama-3.2-3B-Instruct" # "Qwen/Qwen2.5-1.5B-Instruct" 
REWRITE_MODEL = "meta-llama/Llama-3.1-8B-Instruct" #"meta-llama/Llama-3.1-8B-Instruct" Qwen/Qwen2.5-3B-Instruct

STEER_APPLY_MODE = "prefix"  # "prefix" | "step2_window" | "all"
STEER_MIN_TOKEN = 0
STEER_MAX_TOKEN = 128
STEER_WINDOW_CENTER = 64
STEER_WINDOW_PRE = 32
STEER_WINDOW_POST = 32

STEP = 0.5
STEER_LAMBDAS = [i * STEP for i in range(-5, 6)]  # -1.0..1.0
# STEER_LAMBDAS = [1, -1]  # -1.0..1.0

# 或者保留两位，保证打印一致
STEER_LAMBDAS = [0] #[round(x, 2) for x in STEER_LAMBDAS]


applied_sanitized = APPLIED_MODEL.replace("/", "_")
rewrite_sanitized = REWRITE_MODEL.replace("/", "_")

# ==========================================
# 3. 差异化配置
# ==========================================
if EXPERIMENT_MODE == "BASELINE":
    # New mode for GPT Step2 Fix datasets
    VECTOR_DIR = os.path.join(
        "./artifacts",
        f"baseline",
        applied_sanitized + "_applied"
    )
else:
    raise ValueError(f"Unknown EXPERIMENT_MODE: {EXPERIMENT_MODE}")

    
# ==========================================
# 4. 最终路径组装
# ==========================================
task_sanitized = TASKS.replace(",", "_").replace(" ", "_")
BASE_OUTDIR = os.path.join(VECTOR_DIR, task_sanitized)

# ==========================================
# 打印检查 (Optional)
# ==========================================
print(f"Mode: {EXPERIMENT_MODE}")
print(f"Vector Dir:  {VECTOR_DIR}")
print(f"Base Outdir: {BASE_OUTDIR}")
print(f"Batch Size:  {BATCH_SIZE}, Limit: {LIMIT}")

PRETRAINEDS = [
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "meta-llama/Llama-3.2-1B-Instruct",
    # "meta-llama/Llama-3.2-3B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "Qwen/Qwen2.5-0.5B-Instruct",
    # "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-14B-Instruct",
]
MODEL_TO_LAYERS = {
    # "Qwen/Qwen2.5-14B-Instruct": [15, 23, 31],# 
    # "Qwen/Qwen2.5-7B-Instruct": [13, 23, 27],
    # "Qwen/Qwen2.5-1.5B-Instruct": [2,3,4] #[i for i in range(29)],
    # "Qwen/Qwen2.5-0.5B-Instruct": [11, 19, 23],
    # "Qwen/Qwen2.5-3B-Instruct":   [18, 10, 9, 6],
    # "Qwen/Qwen2.5-3B-Instruct":   [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36],
    "Qwen/Qwen2.5-3B-Instruct":   [1],
    # "meta-llama/Llama-3.2-1B-Instruct": [13,14,8] # [i for i in range(17)], # [4, 8,12,16] 
    # "meta-llama/Llama-3.2-3B-Instruct": [6, 16, 20, 22, 23, 24, 25], #[i for i in range(28)], #[16,24,32]
    # "meta-llama/Llama-3.1-8B-Instruct": [8],#[16,24,32]
    # "meta-llama/Llama-3.2-11B-Vision": [8]
}

# 每个新 job 至少需要的空闲显存（MB）。可以视自己的模型大小调。
MIN_FREE_MEM_MB_PER_JOB = 20000  # 例如 20GB
# 每块 GPU 上最多允许同时跑几个进程
MAX_PROCS_PER_GPU = 1

RUNS_JSON = os.path.join(BASE_OUTDIR, "runs.json")

# 全局：job_id -> 记录（会序列化到 runs.json）
RUNS_STATE = {}
# ========= 工具函数：目录 + runs.json 读写 + 显存查询 =========

def ensure_dirs():
    os.makedirs(BASE_OUTDIR, exist_ok=True)


def load_runs_state():
    """从 runs.json 读取历史状态，没有就返回空 dict。"""
    if not os.path.exists(RUNS_JSON):
        return {}
    try:
        with open(RUNS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "jobs" in data and isinstance(data["jobs"], dict):
            return data["jobs"]
        if isinstance(data, dict):
            return data
        return {}
    except Exception as e:
        print(f"[WARN] 读取 {RUNS_JSON} 失败，将从空状态开始：{e}")
        return {}


def save_runs_state():
    """把 RUNS_STATE 写回 runs.json。"""
    ensure_dirs()
    payload = {
        "updated_at": datetime.now().isoformat(),
        "jobs": RUNS_STATE,
    }
    with open(RUNS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def query_gpu_free_mem():
    """
    返回一个 dict: { gpu_id(int): free_mem_mb(int) }.
    依赖 nvidia-smi 命令：
      nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
    如果失败，则假装显存很大（相当于不做限制）。
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        lines = [x.strip() for x in out.splitlines() if x.strip()]
        free_list = []
        for line in lines:
            try:
                free_mb = int(line)
            except ValueError:
                free_mb = 0
            free_list.append(free_mb)

        gpu_free = {}
        for gpu_id, free_mb in enumerate(free_list):
            gpu_free[gpu_id] = free_mb
        return gpu_free

    except Exception as e:
        print(f"[WARN] 无法获取 nvidia-smi 信息，暂时不根据显存限制启动：{e}")
        # 回退：假装每个 GPU 空闲显存都非常大
        return {g: 999999 for g in GPUS}


# ========= Job 定义 =========

class Job:
    def __init__(self, pretrained: str, layer: int, lam: float):
        self.pretrained = pretrained
        self.layer = layer
        self.lam = float(lam)
        self.gpu = None

        self.returncode = None
        self.start_ts = None
        self.end_ts = None
        self.proc = None
        self.status = "pending"  # pending / running / done / failed

        # job_id：用模型名 + layer + lambda 唯一标识
        model_tag = pretrained.replace("/", "_").replace(" ", "_")
        self.job_id = f"{model_tag}__L{layer}__lam{self.lam}"

        tag = "BASELINE" if self.lam == 0.0 else f"lam{str(self.lam).replace('.', 'p')}"
        safe_name = f"{pretrained.split('/')[-1]}_L{layer}_{tag}"
        self.outdir = os.path.join(BASE_OUTDIR, safe_name)
        self.stdout_log = os.path.join(self.outdir, "stdout.log")
        self.stderr_log = os.path.join(self.outdir, "stderr.log")
        self._last_cmd_list = None
    
    def build_cmd(self, gpu_id: int):
        model_tag = self.pretrained.replace("/", "_")
        json_path = (
            os.path.join(VECTOR_DIR,
            f"steering_vector.pt")
        )

        # 1. 增加精度和加速参数
        # 注意：如果你用的是 HuggingFace 后端，推荐加上 flash_attention_2
        model_args = (
            f"pretrained={self.pretrained},"
            # f"device_map=auto,"  # 注意：如果指定了 --device cuda:N，这里最好不要用 auto，或者确保 auto 只看到该卡
            f"dtype=float16,"   # 【加速】使用半精度
            # f"attn_implementation=flash_attention_2," # 【加速】使用 Flash Attention 2
            f"steer_layer={self.layer},"
            f"steer_lambda={self.lam},"
            f"steer_vec_path={json_path},"
            f"steer_apply_mode={STEER_APPLY_MODE},"
            f"steer_min_token={STEER_MIN_TOKEN},"
            f"steer_max_token={STEER_MAX_TOKEN},"
            f"steer_window_center={STEER_WINDOW_CENTER},"
            f"steer_window_pre={STEER_WINDOW_PRE},"
            f"steer_window_post={STEER_WINDOW_POST}"
        )

        # 3. 优化 Batch Size
        # 如果你的显存很大，用 "auto"，如果不确定，先试 "8" 或 "16"

        cmd = [
            "lm_eval",
            "--model", MODEL, # 假设这里是 'hf'
            "--model_args", model_args,
            "--tasks", TASKS,
            "--device", f"cuda:{gpu_id}",
            "--num_fewshot", str(NUM_FEWSHOT), # 确保转为 str
            "--batch_size", BATCH_SIZE,    # 【加速】关键
            "--gen_kwargs", GEN_KWARGS,        # 【控制长度】关键
        ]
        if LIMIT:
            cmd.extend(["--limit", str(LIMIT)])
            
        cmd.extend([
            "--output_path", str(self.outdir),
            "--log_samples",
        ])
        if APPLY_CHAT_TEMPLATE:
            cmd.append("--apply_chat_template")

        self._last_cmd_list = cmd
        # 过滤掉 None 的参数 (例如 limit 为空时)
        
        return cmd

    def cmd_as_str(self):
        return shlex.join(self._last_cmd_list or [])

    def to_record(self):
        duration = None
        if self.start_ts and self.end_ts:
            duration = (self.end_ts - self.start_ts).total_seconds()

        return {
            "job_id": self.job_id,
            "pretrained": self.pretrained,
            "layer": self.layer,
            "lambda": self.lam,
            "is_baseline": bool(self.lam == 0.0),
            "gpu": self.gpu,
            "status": self.status,
            "returncode": self.returncode,
            "start_ts": self.start_ts.isoformat() if self.start_ts else None,
            "end_ts": self.end_ts.isoformat() if self.end_ts else None,
            "duration_sec": duration,
            "outdir": str(self.outdir),
            "stdout_log": str(self.stdout_log),
            "stderr_log": str(self.stderr_log),
            "cmd": self.cmd_as_str(),
        }


# ========= 在指定 GPU 上启动 job =========

def launch_job_on_gpu(job: Job, gpu_id: int):
    """在指定 GPU 上启动一个 job。"""
    global RUNS_STATE

    job.gpu = gpu_id
    os.makedirs(job.outdir, exist_ok=True)

    cmd = job.build_cmd(gpu_id)
    cmd_str = job.cmd_as_str()
    print(f"[LAUNCH] GPU {gpu_id} -> {cmd_str}")

    job.start_ts = datetime.now()
    job.status = "running"

    # 记录 running 状态
    RUNS_STATE[job.job_id] = job.to_record()
    save_runs_state()

    stdout_f = open(job.stdout_log, "w")
    stderr_f = open(job.stderr_log, "w")
    env = os.environ.copy()
    # 注意：这里假定 GPUS 是物理编号，且和 --device cuda:{gpu_id} 一致
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPUS))


    proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f, env=env)
    job.proc = proc
    return proc


# ========= 主调度逻辑 =========
def main():
    global RUNS_STATE
    t0 = time.time()   # ✅ 开始计时
    ensure_dirs()
    RUNS_STATE = load_runs_state()
    if RUNS_STATE:
        print(f"[INFO] 检测到已有 {RUNS_JSON}，将根据历史状态续跑。")
    else:
        print(f"[INFO] 未发现历史记录，将从头开始。")

    # 构造所有作业队列；只依赖 runs.json 判断是否跳过
    queue = []
    for pretrained in PRETRAINEDS:
        valid_layers = MODEL_TO_LAYERS.get(pretrained)
        if valid_layers is None:
            print(f"[WARN] 未找到 {pretrained} 的 layer 配置，跳过")
            continue
        for (L, lam) in itertools.product(valid_layers, STEER_LAMBDAS):
            job = Job(pretrained, L, lam)
            state = RUNS_STATE.get(job.job_id)

            if state and state.get("status") == "done" and state.get("returncode") == 0:
                print(f"[SKIP] {job.job_id} 已在 runs.json 中标记为 done 且 rc=0，跳过。")
                continue

            queue.append(job)
    print(f"[INFO] 总共排队 {len(queue)} 个作业。")
    # running[gpu_id] = [(job, proc), ...]
    running = {g: [] for g in GPUS}

    def handle_sigint(sig, frame):
        print("\n[CTRL-C] 捕获到中断信号，尝试终止子进程 ...")
        for g, lst in running.items():
            for (job, proc) in lst:
                try:
                    proc.terminate()
                    job.status = "failed"
                    job.end_ts = datetime.now()
                    RUNS_STATE[job.job_id] = job.to_record()
                except Exception as e:
                    print(f"[WARN] 终止 GPU {g} 上的进程失败: {e}")
        save_runs_state()
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_sigint)

    # ===== 主循环 =====
    while queue or any(running.values()):
        # 先查询当前每块 GPU 的空闲显存
        gpu_free = query_gpu_free_mem()

        # 尽量在显存足够的 GPU 上启动新作业
        while queue:
            candidate_gpu = None
            best_free_mem = -1

            for g in GPUS:
                procs_on_g = running[g]
                if len(procs_on_g) >= MAX_PROCS_PER_GPU:
                    continue

                free_mb = gpu_free.get(g, 0)
                if free_mb < MIN_FREE_MEM_MB_PER_JOB:
                    continue

                if free_mb > best_free_mem:
                    best_free_mem = free_mb
                    candidate_gpu = g

            # 没有合适的 GPU，就先不启动新作业
            if candidate_gpu is None:
                break

            job = queue.pop(0)
            proc = launch_job_on_gpu(job, candidate_gpu)
            if proc is None:
                # 理论上现在不会返回 None；保留以防你之后加 skip 逻辑
                continue
            running[candidate_gpu].append((job, proc))

        # 轮询所有 GPU 上的进程
        time.sleep(5)
        for g in list(running.keys()):
            new_list = []
            for (job, proc) in running[g]:
                ret = proc.poll()
                if ret is None:
                    # 还在跑
                    new_list.append((job, proc))
                    continue

                # 进程结束
                job.returncode = ret
                job.end_ts = datetime.now()
                job.status = "done" if ret == 0 else "failed"

                RUNS_STATE[job.job_id] = job.to_record()
                save_runs_state()

                # stdout/stderr 写结束标记（可选）
                try:
                    with job.stdout_log.open("a") as f:
                        f.write(f"\n\n[JOB END] returncode={ret} @ {job.end_ts}\n")
                    with job.stderr_log.open("a") as f:
                        f.write(f"\n\n[JOB END] returncode={ret} @ {job.end_ts}\n")
                except Exception:
                    pass

                print(
                    f"[DONE] GPU {g} 完成：L={job.layer}, λ={job.lam}, "
                    f"{'BASELINE' if job.lam==0.0 else 'STEER'}, "
                    f"rc={ret}, status={job.status}"
                )

            running[g] = new_list
    elapsed = time.time() - t0
    print(f"\n[ALL DONE] 全部任务完成。总耗时：{elapsed/60:.1f} min（{elapsed:.0f} s）。汇总见: {RUNS_JSON}")

if __name__ == "__main__":
    main()
