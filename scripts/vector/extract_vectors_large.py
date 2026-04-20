import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from src.eval_utils.prompts import qwen_chat_prompt
import os
import re
# ==========================================
# ===== 引入 steering_vectors 库 ===========
# ==========================================
from steering_vectors import train_steering_vector, SteeringVector

# =====================
# ===== CONFIG ========
# =====================
EXPERIMENT_MODE = "GPT_STEP2_FIX"  # New mode for step2 fix datasets
TARGET_MODEL = "Qwen/Qwen2.5-3B-Instruct"
REWRITE_MODEL = "meta-llama/Llama-3.2-1B-Instruct" # "Qwen/Qwen2.5-7B-Instruct" # 仅在此模式下生效
EXTRACTION_VARIANT = "step2_marker"  # "baseline" | "step2_marker"
STEP2_END_MARKER = "<<STEP2_END>>"

# 层索引配置 (3B range 37)
model_name_to_layer_index = {
    "Qwen/Qwen2.5-3B-Instruct": [i for i in range(37)],
    # "Qwen/Qwen2.5-1.5B-Instruct": [i for i in range(29)],
    # "Qwen/Qwen2.5-0.5B-Instruct": [i for i in range(25)],
    # "meta-llama/Llama-3.2-1B-Instruct": [i for i in range(17)],
    # "meta-llama/Llama-3.2-3B-Instruct": [i for i in range(29)],
}

# 通用配置
NUM_EXAMPLES = None  # Will be set dynamically based on filtered samples

if EXPERIMENT_MODE == "GPT_STEP2_FIX":
    # === New Logic: GPT Step2 Fix with Audit Filter ===
    AUDIT_FILE = "./artifacts/gpt_fix_step2_audit.json"
    DATASET_VARIANT = "fix_step2" # "wait_recompute"  # or "fix_step2"
    MIN_DOC_ID = 300  # Only use samples with id > MIN_DOC_ID
    
    model_name_to_sample_paths = {
        TARGET_MODEL: f"./artifacts/samples_math500_ds2_{DATASET_VARIANT}_gpt.json"
    }
    
    # Output dir will be created after NUM_EXAMPLES is determined
    root_out_dir = None

elif EXPERIMENT_MODE == "GPT_REWRITE":
    DIR_PATH = "./gpt_rewrites_unified_new"
    PROMPT_STYLE = "old"  # 核心变量：仅在此模式下生效
    
    # 构造路径
    REWRITEEN_SAMPLE_PATH = os.path.join(DIR_PATH, TARGET_MODEL.replace("/", "_"))
    
    model_name_to_sample_paths = {
        TARGET_MODEL: os.path.join(REWRITEEN_SAMPLE_PATH, f"rewritten_{PROMPT_STYLE}.json")
    }
    
    # 输出目录
    NUM_EXAMPLES = 50  # Fixed for non-GPT_STEP2_FIX modes
    root_out_dir = Path(REWRITEEN_SAMPLE_PATH) / f"vectors_{NUM_EXAMPLES}_{PROMPT_STYLE}"

elif EXPERIMENT_MODE == "LARGE_MODEL":
    # === 逻辑 2: Large Model Rewrites (Qwen 0.5B) ===
    DIR_PATH = "./large_model_rewrites_unified_new"
    
    # 构造路径
    REWRITEEN_SAMPLE_PATH = os.path.join(DIR_PATH, TARGET_MODEL.replace("/", "_"))
    
    model_name_to_sample_paths = {
        TARGET_MODEL: os.path.join(REWRITEEN_SAMPLE_PATH, f"{REWRITE_MODEL.replace('/', '_')}_paired_responses.json"),
    }
    
    # 输出目录
    NUM_EXAMPLES = 50  # Fixed for non-GPT_STEP2_FIX modes
    root_out_dir = Path(REWRITEEN_SAMPLE_PATH) / f"vectors_{NUM_EXAMPLES}_paired_{REWRITE_MODEL.replace('/', '_')}"

else:
    raise ValueError(f"Unknown EXPERIMENT_MODE: {EXPERIMENT_MODE}")

# 创建输出目录（如果已确定）
if root_out_dir is not None:
    root_out_dir.mkdir(exist_ok=True, parents=True)

# ==========================================
# 打印检查 (Optional)
# ==========================================
print(f"Current Mode: {EXPERIMENT_MODE}")
print(f"Target Model: {TARGET_MODEL}")
print(f"Sample Path:  {model_name_to_sample_paths[TARGET_MODEL]}")
print(f"Output Dir:   {root_out_dir}")
print(f"Extraction Variant: {EXTRACTION_VARIANT}")

# =====================
# ===== HELPERS =======
# =====================

def get_exact_match(ex: dict):
    if "exact_match" in ex:
        try: return float(ex["exact_match"])
        except: pass
    for k in ["metrics", "results", "scores"]:
        if k in ex and isinstance(ex[k], dict) and "exact_match" in ex[k]:
            try: return float(ex[k]["exact_match"])
            except: pass
    return None

def load_samples(path: str):
    path = str(path)
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for k in ["samples", "instances", "data"]:
                if k in obj and isinstance(obj[k], list): return obj[k]
        return obj

def load_audit_incorrect_ids(audit_path: str):
    """Load doc_ids with judge_step2_correct=false from audit file"""
    with open(audit_path, "r", encoding="utf-8") as f:
        audit_data = json.load(f)
    incorrect_ids = {
        item["doc_id"] 
        for item in audit_data 
        if item.get("judge_step2_correct") is False
    }
    return incorrect_ids

def insert_step2_end_marker(text: str, marker: str = STEP2_END_MARKER) -> str:
    """
    Insert marker before Step 3 if possible, otherwise append at the end.
    This keeps Step2 boundary explicit in both pos/neg responses.
    """
    if not isinstance(text, str):
        return ""
    if marker in text:
        return text

    step3_patterns = [
        r"(?im)^\s*step\s*3\b",
        r"(?im)^\s*\*\*\s*step\s*3\b",
        r"(?im)^\s*3\s*[\)\.:：、]",
    ]

    insert_at = None
    for pat in step3_patterns:
        m = re.search(pat, text)
        if m:
            insert_at = m.start()
            break

    if insert_at is None:
        return text.rstrip() + f"\n{marker}"

    left = text[:insert_at].rstrip()
    right = text[insert_at:].lstrip("\n")
    return f"{left}\n{marker}\n{right}"

def trim_text_to_marker(text: str, marker: str = STEP2_END_MARKER):
    """
    Keep text up to marker (inclusive). If marker missing, append marker at tail.
    After trimming, read_token_index=-1 corresponds to marker location.
    """
    idx = text.find(marker)
    if idx == -1:
        return text.rstrip() + f"\n{marker}", False
    end = idx + len(marker)
    return text[:end], True

# =====================
# GPT_STEP2_FIX PRE-PROCESSING
# =====================

gpt_step2_selected_ids = None  # Will hold selected IDs for GPT_STEP2_FIX mode

if EXPERIMENT_MODE == "GPT_STEP2_FIX":
    # Pre-load audit and filter samples to determine NUM_EXAMPLES before main loop
    sample_path = model_name_to_sample_paths[TARGET_MODEL]
    samples = load_samples(sample_path)
    incorrect_ids = load_audit_incorrect_ids(AUDIT_FILE)
    print(f"[Pre-processing] Loaded {len(incorrect_ids)} incorrect doc_ids from audit file")
    
    # Build dict by doc.id with id > 300 filter
    by_id = {}
    for ex in samples:
        doc = ex.get("doc", {})
        doc_id = doc.get("id", -1)
        if doc_id in incorrect_ids and doc_id > 300:  # Filter by id > 300
            by_id[doc_id] = ex
    
    doc_ids = sorted(by_id.keys())
    gpt_step2_selected_ids = doc_ids
    
    # Update NUM_EXAMPLES to actual count
    NUM_EXAMPLES = len(gpt_step2_selected_ids)
    print(f"[Pre-processing] Found {NUM_EXAMPLES} samples with id > 300 and judge_step2_correct=false")
    
    # Create output directory with actual NUM_EXAMPLES
    out_suffix = "_step2_marker" if EXTRACTION_VARIANT == "step2_marker" else ""
    root_out_dir = Path(f"./artifacts/vectors_{NUM_EXAMPLES}_ds2_{DATASET_VARIANT}_incorrect_only_id300+{out_suffix}")
    root_out_dir.mkdir(exist_ok=True, parents=True)
    print(f"[Pre-processing] Output directory: {root_out_dir}")

# =====================
# ===== MAIN LOOP =====
# =====================

for model_name, layer_list in model_name_to_layer_index.items():
    print(f"\n========== Processing model: {model_name} ==========")

    model_tag = model_name.replace("/", "_")
    model_tag += "_applied"
    model_out_dir = root_out_dir / model_tag
    model_out_dir.mkdir(exist_ok=True)

    # 1. 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,      
        device_map="auto",
    ).eval()

    # 2. 准备数据
    sample_path = model_name_to_sample_paths[model_name]
    samples = load_samples(sample_path)
    
    # Load audit filter if in GPT_STEP2_FIX mode
    if EXPERIMENT_MODE == "GPT_STEP2_FIX":
        incorrect_ids = load_audit_incorrect_ids(AUDIT_FILE)
        
        # Build dict by doc.id with id > 300 filter
        by_id = {}
        for ex in samples:
            doc = ex.get("doc", {})
            doc_id = doc.get("id", -1)
            if doc_id in incorrect_ids and doc_id > 300:  # Filter by id > 300
                by_id[doc_id] = ex
        
        selected_ids = gpt_step2_selected_ids
        print(f"Selected {len(selected_ids)} samples for steering.")
        
        # Construct training pairs: pos_response (corrected) vs neg_response (original)
        training_samples = []
        marker_hits = 0
        for did in selected_ids:
            ex = by_id[did]
            question = ex["doc"]["question"]
            prompt = qwen_chat_prompt(question)
            pos_response = ex.get("pos_response", "")
            neg_response = ex.get("neg_response", "")
            if pos_response and neg_response:
                if EXTRACTION_VARIANT == "step2_marker":
                    pos_response = insert_step2_end_marker(pos_response)
                    neg_response = insert_step2_end_marker(neg_response)

                    pos_text, pos_has_marker = trim_text_to_marker(prompt + pos_response)
                    neg_text, neg_has_marker = trim_text_to_marker(prompt + neg_response)
                    marker_hits += int(pos_has_marker) + int(neg_has_marker)
                    training_samples.append((pos_text, neg_text))
                else:
                    training_samples.append((prompt + pos_response, prompt + neg_response))
        
        print(f"Built {len(training_samples)} training pairs from filtered samples.")
        if EXTRACTION_VARIANT == "step2_marker":
            print(f"Marker hits in original responses: {marker_hits}")
    
    else:
        # Original logic for other modes
        by_id = {ex["doc_id"]: ex for ex in samples if "doc_id" in ex}
        doc_ids = sorted(by_id.keys())
        
        # 筛选准确率 1.0 的样本（从后往前）
        selected_ids = []
        for did in reversed(doc_ids):
            if get_exact_match(by_id[did]) == 1.0:
                selected_ids.append(did)
                if len(selected_ids) >= NUM_EXAMPLES: break
        selected_ids = list(reversed(selected_ids))

        print(f"Selected {len(selected_ids)} samples for steering.")

        # 3. 构造正负样本对 (positive, negative)
        # 根据你的逻辑：rewritten 是更好的行为 (Positive)，original 是之前的行为 (Negative)
        training_samples = []
        marker_hits = 0
        for did in selected_ids:
            ex = by_id[did]
            prompt = qwen_chat_prompt(ex["doc"]["question"])
            pos_response = ex["resp_after"]
            neg_response = ex["resp_before"]
            if EXTRACTION_VARIANT == "step2_marker":
                pos_response = insert_step2_end_marker(pos_response)
                neg_response = insert_step2_end_marker(neg_response)

                pos_text, pos_has_marker = trim_text_to_marker(prompt + pos_response)
                neg_text, neg_has_marker = trim_text_to_marker(prompt + neg_response)
                marker_hits += int(pos_has_marker) + int(neg_has_marker)
                training_samples.append((pos_text, neg_text))
            else:
                training_samples.append((prompt + pos_response, prompt + neg_response))

        if EXTRACTION_VARIANT == "step2_marker":
            print(f"Marker hits in original responses: {marker_hits}")

    # 4. 训练转向向量 (使用文档中的 train_steering_vector)
    if training_samples:
        print(f"  → Training steering vector for layers: {layer_list}")
        
        # 按照文档参数调用
        read_token_index = -1

        steering_vector = train_steering_vector(
            model=model,
            tokenizer=tokenizer,
            training_samples=training_samples,
            layers=layer_list,
            layer_type="decoder_block", # Qwen 默认使用 decoder 结构
            move_to_cpu=True,           # 节省显存，将结果存放在 CPU
            read_token_index=read_token_index,
            show_progress=True,
            batch_size=1                # 如果显存充足可以调大
        )

        # 5. 保存结果
        # 注意：使用 torch.save 存储 SteeringVector 对象
        save_path = model_out_dir / "steering_vector.pt"
        torch.save(steering_vector, save_path)
        
        print(f"  ✔ Successfully saved SteeringVector to {save_path}")
        # steering_vector.layer_activations 是一个 dict {layer_idx: tensor}
        print(f"     Layers in object: {list(steering_vector.layer_activations.keys())}")

        # Calculate and save norms
        norms = {}
        # Check if layer_activations acts as a dict (keys are layer indices)
        for layer_idx, vec in steering_vector.layer_activations.items():
            # vec is likely a tensor of shape [hidden_dim] or [1, hidden_dim]
            norm_val = vec.norm().item()
            norms[layer_idx] = norm_val
            
        norms_path = model_out_dir / "vector_norms.json"
        with open(norms_path, "w") as f:
            json.dump(norms, f, indent=2)
        print(f"  ✔ Saved vector norms to {norms_path}")


    del model
    torch.cuda.empty_cache()

print("\nAll tasks completed.")