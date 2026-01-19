import os
import sys
import json
import argparse
import torch
import pandas as pd
import numpy as np
import gc
import lm_eval.api.registry
# Add paths to sys.path to allow importing modules from sibling directories
# Assuming cr_evaluator_avg.py is in Infer_Comm/evaluation/param_search/
# We need to reach Infer_Comm/evaluation/lm_eval/
sys.path.append("/root/workspace")

# import Infer_Comm.evaluation.lm_eval.lm_wrapper as lm_wrapper
# import Infer_Comm.evaluation.lm_eval.lm_evaluator as lm_evaluator
from Infer_Comm.evaluation.lm_eval import lm_wrapper, lm_evaluator


# 硬编码路径 (参考自 custom_accuracy.py)
BASE_MODEL_PATH = "/root/workspace/models"
BASE_CONFIG_PATH = "/root/workspace/Infer_Comm/duo_config"

TASK_TO_CHAT_TEMPLATE = {
    "longbench_lcc": False,
    "longbench_lcc_e": False,
    "longbench_repobench-p": False,
    "longbench_repobench-p_e": False,
    "longbench_multi_news": True,
    "longbench_multi_news_e": True,
    "longbench_gov_report": True,
    "longbench_gov_report_e": True,
    "longbench_2wikimqa": True,
    "longbench_2wikimqa_e": True,
    "longbench_hotpotqa": True,
    "longbench_hotpotqa_e": True,
    "longbench_qasper": True,
    "longbench_qasper_e": True,
    "longbench_multifieldqa_en": True,
    "longbench_multifieldqa_en_e": True,
    "gsm8k_cot_llama": True,
    "gsm8k_cot": False,
    "mbpp_instruct": True,
    "humaneval_instruct": True,
}

def evaluate_config(model_name, tasks, params, device="cuda"):
    # Load scores
    scores_path = f"{BASE_CONFIG_PATH}/{model_name}_scores.csv"
    if os.path.exists(scores_path):
        df = pd.read_csv(scores_path, header=None).dropna()
        scores = torch.tensor(df.values, dtype=torch.float32)
    else:
        print(f"Warning: Scores file not found at {scores_path}")
        scores = None

    compression_ratio_list = []
    
    # Construct model_args from params
    model_args = {
        "pretrained": f"{BASE_MODEL_PATH}/{model_name}",
        "device_map": "auto",
        "parallelize": True,
        "attn_implementation": "flash_attention_2",
        
        "transform_type": params.get("transform_type", "none"),
        "scores": scores,
        "heads_selection": params.get("heads_selection", 1.0),
        "high_key_max_value": params.get("high_key_max_value", 10),
        "high_value_max_value": params.get("high_value_max_value", 6),
        "low_key_max_value": params.get("low_key_max_value", 8),
        "low_value_max_value": params.get("low_value_max_value", 6),
        "axis_key": params.get("axis_key", [2]),
        "axis_value": params.get("axis_value", [1, 3]),
        
        "cache_type": "custom", 
        "comp_cr": True,
        "cr_list": compression_ratio_list,
    }

    print(f"Running evaluation for config ID: {params.get('config_id', 'N/A')} on tasks: {tasks}")
    
    results = lm_evaluator.simple_evaluate(
        model="my_custom_model", 
        model_args=model_args,
        tasks=tasks,
        device=device,
        batch_size="1",
        apply_chat_template=TASK_TO_CHAT_TEMPLATE,
        confirm_run_unsafe_code=True,
        gen_kwargs={"max_gen_toks": 2},
    )

    if not compression_ratio_list:
        print("Warning: No compression ratios recorded.")
        return 0.0
        
    # Save compression_ratio_list using config_id
    config_id = params.get('config_id', 'unknown')
    save_path = os.path.join(".", f"{config_id}.json")

    # Ensure all elements are standard floats for JSON serialization
    serializable_list = [float(x) for x in compression_ratio_list]
    with open(save_path, 'w') as f:
        json.dump(serializable_list, f)
    print(f"Saved compression ratio list to {save_path}")

    avg_cr = np.mean(compression_ratio_list)
    print(f"Config ID {params.get('config_id', 'N/A')} - Average Compression Ratio: {avg_cr:.4f}")
    del results, compression_ratio_list
    gc.collect()
    torch.cuda.empty_cache()
    
    return float(avg_cr)

parser = argparse.ArgumentParser(description="Evaluate compression ratio using lm_eval")
parser.add_argument("--model_name", type=str, default="Qwen2.5-32B-Instruct", help="Model name")
parser.add_argument("--tasks", type=str, nargs='+', default=["longbench_qasper"], help="List of tasks to evaluate")
parser.add_argument("--json_path", type=str, required=True, help="Path to input JSON parameter file")

args = parser.parse_args()

# Check if JSON file exists
if not os.path.exists(args.json_path):
    print(f"Error: JSON file not found at {args.json_path}")

# Read JSON parameters
print(f"Reading parameters from {args.json_path}...")
with open(args.json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Iterate through configurations
if isinstance(data, list):
    configs = data
elif isinstance(data, dict):
    configs = [data]
else:
    print("Error: JSON content must be a list or a dict")

for i, params in enumerate(configs):
    print(f"Processing config {i+1}/{len(configs)}: ID={params.get('config_id', 'N/A')}")
    avg_cr = evaluate_config(args.model_name, args.tasks, params)
    params['cr'] = avg_cr


# Save results back to JSON
print(f"Saving results to {args.json_path}...")
with open(args.json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("Done.")
