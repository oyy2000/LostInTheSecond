# prm_utils.py
import re


# ---------------------------------------------------------------------------
# Tokenizer-based chat template (canonical, preferred for all new scripts)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def build_chat_prompt_from_tokenizer(
    tokenizer,
    question: str,
    system: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Build a chat prompt using the tokenizer's apply_chat_template.

    This ensures the prompt exactly matches the format the model was trained on,
    including any special tokens or whitespace conventions.

    Returns the prompt string ending with the assistant turn start, ready for
    the model to continue generating.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question.strip()},
    ]
    # tokenize=False → returns a string
    # add_generation_prompt=True → appends the assistant turn header
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


# ---------------------------------------------------------------------------
# Legacy hand-crafted templates (kept for backward compatibility)
# ---------------------------------------------------------------------------

def qwen_chat_prompt(
    question: str,
    system: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
) -> str:
    """
    Canonical Qwen chat template (string form).
    No dependency on lm-eval arg_0.
    """
    return (
        "<|im_start|>system\n"
        f"{system}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Solve the following math problem. Present the final answer in the format: Final Answer: \\boxed{your_answer}.\n"
        f"Prolbem: {question}\n"
        "Answer:\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def qwen_chat_prompt_no_special_token(
    question: str,
    system: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
) -> str:
    """
    Canonical Qwen chat template (string form).
    No dependency on lm-eval arg_0.
    """
    return (
        "Solve the following math problem. Present the final answer in the format: Final Answer: \\boxed{your_answer}.\n"
        f"Prolbem: {question}\n"
        "Answer:"
    )

def qwen_chat_prompt_old(
    question: str,
    system: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
) -> str:
    """
    Canonical Qwen chat template (string form).
    No dependency on lm-eval arg_0.
    """
    return (
        "<|im_start|>system\n"
        f"{system}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Q: {question}\n"
        "A: Let's think step by step.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def split_double_newline(text):
    """策略1: 严格双换行"""
    if not text: return []
    parts = text.split('\n\n')
    steps = []
    for i, p in enumerate(parts):
        if not p.strip(): continue 
        if i < len(parts) - 1:
            steps.append(p + '\n\n')
        else:
            steps.append(p)
    return steps

def split_single_newline_robust(text):
    """策略2: 鲁棒单换行 (自动把 \n\n 粘回去)"""
    if not text: return []
    # 使用捕获组 (\n+) 切分，保留分隔符
    parts = re.split(r'(\n+)', text)
    steps = []
    for i in range(0, len(parts), 2):
        step = parts[i]
        if i + 1 < len(parts):
            step += parts[i+1] # 把分隔符加回 Step 末尾
        if step.strip():
            steps.append(step)
    return steps

def split_auto(text):
    """默认策略: 优先双换行"""
    if "\n\n" in text:
        return split_double_newline(text)
    return split_single_newline_robust(text)

import os
import json
import glob
import numpy as np
from transformers import AutoTokenizer

MODEL_MAP = {
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Llama-3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
}

def get_model_path(folder_model_name):
    """尝试从文件夹名推断 HF 模型路径"""
    if folder_model_name in MODEL_MAP:
        return MODEL_MAP[folder_model_name]
    # 简单的 heuristic: 将下划线转回斜杠 (例如 Qwen_Qwen -> Qwen/Qwen)
    if "Qwen" in folder_model_name and "_" in folder_model_name:
         return folder_model_name.replace("_", "/", 1) # 只替换第一个
    return folder_model_name

def process_single_file(jsonl_path, tokenizer):
    """处理单个 JSONL 文件，返回统计数据列表"""
    stats = {
        "Y": [],
        "tokens_per_step_avg_single": [],
        "tokens_per_step_avg_double": [],
        # 策略对比数据
        "steps_count_double": [],        
        "steps_count_single": [],
        "total_tokens_double": [],
        "total_tokens_single": [],
        "text_split_double": [],
        "text_split_single": [],
    }
    
    try:
        data = [json.loads(l) for l in open(jsonl_path, 'r', encoding='utf-8')]
    except Exception as e:
        print(f"Error reading {jsonl_path}: {e}")
        return None

    for d in data:
        if d.get("filter") == "strict-match": continue
        
        # 获取 CoT
        try:
            cot = d["resps"][0][0].strip()
        except: continue
        
        is_correct = int(d.get("exact_match", 0))
        
        # --- 计算 Strategies ---
        steps_dbl = split_double_newline(cot)
        tokens_dbl = [len(tokenizer.encode(s, add_special_tokens=False)) for s in steps_dbl]
        
        steps_sgl = split_single_newline_robust(cot)
        tokens_sgl = [len(tokenizer.encode(s, add_special_tokens=False)) for s in steps_sgl]

        # --- 存入 Stats ---
        stats["Y"].append(is_correct)
        
        # Comparison Metrics
        stats["steps_count_double"].append(len(steps_dbl))
        stats["total_tokens_double"].append(sum(tokens_dbl))
        stats["text_split_double"].append(steps_dbl)
        stats["tokens_per_step_avg_double"].append(np.mean(tokens_dbl) if tokens_dbl else 0)
        
        stats["steps_count_single"].append(len(steps_sgl))
        stats["total_tokens_single"].append(sum(tokens_sgl))
        stats["text_split_single"].append(steps_sgl)
        stats["tokens_per_step_avg_single"].append(np.mean(tokens_sgl) if tokens_sgl else 0)
        
    return stats

def scan_and_process(base_dir):
    """扫描目录，加载 Tokenizer，执行任务"""
    results = {} # {model: {layer: {lam: stats}}}
    
    # 1. 扫描文件夹结构
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return {}

    entries = sorted(os.listdir(base_dir))
    tasks = []

    print(f"Scanning {base_dir} ...")
    for entry in entries:
        full_path = os.path.join(base_dir, entry)
        if not os.path.isdir(full_path): continue
        
        # 解析文件夹名: Model_Name_Lxx_lamxx
        parts = entry.split('_')
        if len(parts) < 3: continue
        
        lam_str = parts[-1]   # e.g., lam1
        layer_str = parts[-2] # e.g., L10
        model_key = "_".join(parts[:-2]) # e.g., Qwen_Qwen2.5...
        
        # 寻找 JSONL 文件
        # 通常在 full_path 下面还有一层或者直接是 jsonl
        # 这里假设 logic: full_path/subdir/samples_*.jsonl
        jsonl_candidates = glob.glob(f"{full_path}/**/samples_*.jsonl", recursive=True)
        if not jsonl_candidates:
            continue
        
        # 取最新的一个
        jsonl_file = sorted(jsonl_candidates)[-1]
        
        tasks.append({
            "model_key": model_key,
            "layer": layer_str,
            "lam": lam_str,
            "jsonl": jsonl_file
        })

    # 2. 按模型分组执行 (为了复用 Tokenizer)
    tasks_by_model = {}
    for t in tasks:
        tasks_by_model.setdefault(t["model_key"], []).append(t)
        
    print(f"Found {len(tasks)} tasks across {len(tasks_by_model)} models.")

    for model_key, model_tasks in tasks_by_model.items():
        hf_path = get_model_path(model_key)
        print(f"\n>>> Loading Tokenizer for {model_key} ({hf_path})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load tokenizer for {model_key}: {e}")
            continue
            
        for t in model_tasks:
            print(f"   Processing {t['layer']} - {t['lam']} ...")
            file_stats = process_single_file(t['jsonl'], tokenizer)
            
            if file_stats:
                # 存入结果结构
                results.setdefault(model_key, {})
                results[model_key].setdefault(t['layer'], {})
                results[model_key][t['layer']][t['lam']] = file_stats

    return results
