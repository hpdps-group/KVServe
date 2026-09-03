import os
import sys
import torch
import pandas as pd
import numpy as np
import pickle
import gc
import argparse
import random
import lm_eval
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from collections import deque
from contextlib import contextmanager
from lm_eval.tasks import TaskManager, get_task_dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from offline_search.evaluation.lm_eval import lm_wrapper, lm_evaluator

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
    "longbench_trec": False,
    "gsm8k_cot_llama": True,
    "gsm8k_cot": False,
    "mbpp_instruct": True,
    "humaneval_instruct": True,
}

@contextmanager
def suppress_fd_stderr(enabled=True):
    """
    Suppress stderr at the OS file-descriptor level.

    Set enabled=False to bypass suppression while keeping the same context manager API.
    """
    if not enabled:
        # Leave stderr unchanged.
        yield
        return

    devnull = os.open(os.devnull, os.O_WRONLY)
    original_stderr_fd = os.dup(2)
    try:
        sys.stderr.flush()
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(original_stderr_fd, 2)
        os.close(devnull)
        os.close(original_stderr_fd)
 

class AccuracyEvaluator:
    def __init__(
        self,
        model_name="Meta-Llama-3.1-8B-Instruct",
        tasks=["longbench_hotpotqa"],
        limit=100,
        batch_size=1,
        device="cuda",
        random_seed=42,
        base_model_path=None,
        base_config_path=None,
        attn_implementation="sdpa",
    ):
        print(f"Loading model {model_name} and preparing data... (This happens only once)")
        self.device = device
        self.model_name = model_name
        self.tasks = tasks
        self.limit = limit
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.attn_implementation = attn_implementation
        if base_model_path is None or base_config_path is None:
            raise ValueError("base_model_path and base_config_path must be provided by the search script.")
        self.base_model_path = base_model_path
        self.base_config_path = base_config_path

        # Precompute uniformly sampled evaluation document indices.
        self.samples = {}
        try:
            task_manager = TaskManager()
            all_tasks = get_task_dict(self.tasks, task_manager)
            
            # Seed sampling so repeated searches evaluate the same examples.
            random.seed(self.random_seed)

            for task_name, task_obj in all_tasks.items():
                if hasattr(task_obj, 'test_docs') and task_obj.has_test_docs():
                    docs = task_obj.test_docs()
                elif hasattr(task_obj, 'validation_docs') and task_obj.has_validation_docs():
                    docs = task_obj.validation_docs()
                else:
                    docs = task_obj.training_docs()
                
                # Determine dataset size.
                try:
                    n_samples = len(docs)
                except:
                    # Some task iterables do not expose len(); materialize them as a fallback.
                    all_docs = list(docs)
                    n_samples = len(all_docs)
                
                if n_samples <= self.limit:
                    self.samples[task_name] = list(range(n_samples))
                else:
                    # Sample one document from each evenly spaced interval.
                    step = n_samples / self.limit
                    selected = []
                    for i in range(self.limit):
                        start = int(i * step)
                        end = int((i + 1) * step)
                        if end <= start:
                            end = start + 1
                        if end > n_samples:
                            end = n_samples
                        
                        # Draw one random index from the current interval.
                        if start < end:
                            selected.append(random.randrange(start, end))
                        else:
                            selected.append(start)
                    self.samples[task_name] = selected
            print(f"Selected samples indices for tasks: { {k: len(v) for k,v in self.samples.items()} }")
        except Exception as e:
            print(f"Warning: Failed to prepare samples in __init__: {e}")
            # Fallback logic or just re-raise if critical
            raise e

        # Load per-head scores used by the custom cache policy.
        scores_path = f"{self.base_config_path}/{model_name}_scores.csv"
        df = pd.read_csv(scores_path, header=None).dropna()
        self.scores = torch.tensor(df.values, dtype=torch.float32)
        # Evaluate the baseline model once to normalize custom-cache scores.
        model_args = {
            "pretrained": f"{self.base_model_path}/{self.model_name}",
            "device_map": "auto",
            "parallelize": True,
            "attn_implementation": self.attn_implementation,
            "cache_type": "default",
        }
        with suppress_fd_stderr():
            default_results = lm_evaluator.simple_evaluate(
                model="my_custom_model",
                model_args=model_args,
                tasks=self.tasks,
                device="cuda",
                batch_size=self.batch_size,
                apply_chat_template=TASK_TO_CHAT_TEMPLATE,
                confirm_run_unsafe_code=True,

                samples=self.samples,
            )
        default_values = []
        for results in default_results:
            for task_name, metrics in results['results'].items():
                # print(f"Task: {task_name}")
                # print(metrics)
                for metric_name, value in metrics.items():
                    if "score" in metric_name and not "_stderr" in metric_name:
                        # Keep a stable precision for downstream ratio calculations.
                        print(f"  {metric_name}: {value}")
                        default_values.append(round(value, 4))
                        # break
        self.default_scores = np.array(default_values)
        self.valid_baseline_mask = np.isfinite(self.default_scores) & (np.abs(self.default_scores) > 1e-12)
        if not self.valid_baseline_mask.any():
            raise RuntimeError(
                "All sampled baseline metrics are zero; increase dataset_limit or change the sampling seed"
            )
        del default_results
        gc.collect()
        torch.cuda.empty_cache()        

    def evaluate(self, params):
        """
        Args:
            params: Search configuration containing heads_selection, max-value
                bounds, axis settings, and transform_type.
        """

        # Configure the model to run with the custom cache policy.
        model_args = {
            "pretrained": f"{self.base_model_path}/{self.model_name}",
            "device_map": "auto",
            "parallelize": True,
            "attn_implementation": self.attn_implementation,

            "transform_type": params["transform_type"],
            "scores": self.scores,
            "heads_selection": params["heads_selection"],
            "high_key_max_value": params["high_key_max_value"],
            "high_value_max_value": params["high_value_max_value"],
            "low_key_max_value": params["low_key_max_value"],
            "low_value_max_value": params["low_value_max_value"],
            "axis_key": list(params["axis_key"]),
            "axis_value": list(params["axis_value"]),

            "cache_type": "custom",
        }
        with suppress_fd_stderr():
            custom_results = lm_evaluator.simple_evaluate(
                model="my_custom_model",
                model_args=model_args,
                tasks=self.tasks,
                device="cuda",
                batch_size=self.batch_size,
                
                # Optional lm-eval settings.
                apply_chat_template=TASK_TO_CHAT_TEMPLATE,
                confirm_run_unsafe_code=True,
                samples=self.samples,
                # num_fewshot=5,
            )

        custom_values = []
        for results in custom_results:
            for task_name, metrics in results['results'].items():
                # print(f"Task: {task_name}")
                for metric_name, value in metrics.items():
                    if "score" in metric_name and not "_stderr" in metric_name:
                        # Keep a stable precision for downstream ratio calculations.
                        # print(f"  {metric_name}: {value:.4f}")
                        custom_values.append(round(value, 4))
                        # break
        custom_scores = np.array(custom_values)
        if custom_scores.shape != self.default_scores.shape:
            raise RuntimeError(
                f"Metric count changed between baseline ({self.default_scores.size}) "
                f"and custom cache ({custom_scores.size})"
            )
        valid_custom = np.isfinite(custom_scores[self.valid_baseline_mask])
        if not valid_custom.all():
            raise RuntimeError("Custom-cache evaluation returned a non-finite task metric")
        # A zero baseline metric cannot define relative accuracy, so exclude it
        # rather than allowing 0/0 to poison the Bayesian optimizer with NaN.
        avg_score = (
            custom_scores[self.valid_baseline_mask]
            / self.default_scores[self.valid_baseline_mask]
        ).mean() * 100
        print(f"Custom scores: {custom_scores}")
        print(f"Default scores: {self.default_scores}")
        print(f"Avg score: {avg_score}")
        del custom_results
        gc.collect()
        torch.cuda.empty_cache()
        return round(avg_score, 2)

# params = {
#     "transform_type": "none",
#     "heads_selection": 0.9,
#     "high_key_max_value": 8,
#     "high_value_max_value": 10,
#     "low_key_max_value": 8,
#     "low_value_max_value": 6,
#     "axis_key": [2],
#     "axis_value": [1, 3],
# }
# eva = AccuracyEvaluator(
#     tasks=["longbench_2wikimqa"],
#     model_name="Qwen2.5-7B-Instruct",
#     limit=5,
#     base_model_path="/root/data/models",
#     base_config_path="/root/workspaces/KVServe_opensourced/kvserve_v1/offline_search/duo_config",
# )
# eva.evaluate(params)
# Potential pruning extension:
# first evaluate accuracy on half of the sampled dataset; if the score is above
# tolerance, continue with the remaining half, otherwise consider pruning.
# Code: mbpp_instruct(250)
# Math: gsm8k_cot_llama/gsm8k_cot(200/250)
# QA: qasper/2wikimqa
# Summarization: multi_news/gov_report
