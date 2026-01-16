#!/usr/bin/env python3
"""
Test script for Prefill-Decode separation simulator.

Usage:
    python test/test_simulator.py tp1 --num-requests 50 --request-rate 1 --lmeval-task longbench_qasper
    python test/test_simulator.py tp2 --num-requests 100 --request-rate 2
"""

import argparse
import asyncio
import os
import sys
from typing import List, Optional

# Add project root to Python path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Disable proxy to avoid connection issues with HuggingFace/local models
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)


def load_lmeval_prompts(task_name: str, num_samples: int = 50) -> List[str]:
    """
    Load prompts from lm-eval-harness task.
    
    Args:
        task_name: Name of the lm-eval task (e.g., 'longbench_qasper')
        num_samples: Number of samples to load
    
    Returns:
        List of prompt strings
    """
    try:
        import lm_eval
        import lm_eval.tasks
    except ImportError:
        raise ImportError("lm-eval-harness not installed. Install with: pip install lm-eval")
    
    print(f"[INFO] Loading lm-eval task: {task_name}")
    
    # Try newer API first
    try:
        # Try with list argument
        try:
            task_dict = lm_eval.tasks.get_task_dict([task_name])
        except TypeError:
            # Fallback for older versions that don't require list
            task_dict = lm_eval.tasks.get_task_dict(task_name)
        
        task_obj = task_dict[task_name]
        
        # Check if it's a class or instance
        if isinstance(task_obj, type):
            # It's a class, instantiate it
            task = task_obj()
        else:
            # It's already an instance
            task = task_obj
    except (AttributeError, TypeError) as e:
        # Fallback to older API
        print(f"[INFO] Trying fallback API due to: {e}")
        try:
            task = lm_eval.tasks.get_task(task_name)
        except AttributeError:
            raise RuntimeError(f"Could not load task '{task_name}' with any known lm-eval API")
    
    # Get dataset
    if hasattr(task, 'dataset'):
        dataset = task.dataset
    elif hasattr(task, 'get_dataset'):
        dataset = task.get_dataset()
    else:
        raise RuntimeError(f"Task '{task_name}' has no dataset attribute or get_dataset method")
    
    # Try different split names
    split_names = ['test', 'validation', 'train']
    split_data = None
    
    for split_name in split_names:
        try:
            if isinstance(dataset, dict):
                if split_name in dataset:
                    split_data = dataset[split_name]
                    print(f"[INFO] Using split: {split_name}")
                    break
            else:
                split_data = dataset[split_name]
                print(f"[INFO] Using split: {split_name}")
                break
        except (KeyError, AttributeError):
            continue
    
    if split_data is None:
        available_splits = list(dataset.keys()) if isinstance(dataset, dict) else "unknown"
        raise RuntimeError(
            f"No available split for lm-eval task '{task_name}'. "
            f"Tried: {split_names}, Available: {available_splits}. "
            f"Debug: dataset type = {type(dataset)}"
        )
    
    # Extract prompts
    prompts = []
    for i, example in enumerate(split_data):
        if i >= num_samples:
            break
        
        # Try different field names for prompt/context
        prompt_text = None
        for field in ['context', 'input', 'question', 'text', 'prompt']:
            if field in example:
                prompt_text = example[field]
                break
        
        if prompt_text is None:
            print(f"[WARNING] Could not find prompt field in example {i}, keys: {example.keys()}")
            continue
        
        prompts.append(str(prompt_text))
    
    print(f"[INFO] Loaded {len(prompts)} prompts from '{task_name}'")
    return prompts


async def worker_prefill(
    tp: int,
    intermediate_file: str,
    num_requests: int,
    request_rate_rps: float,
    lmeval_task: Optional[str] = None,
    compression_config=None,
):
    """Run prefill stage"""
    from kvserve.simulator.simulator_backend import SimulatorBackend
    
    print(f"[INFO] [Prefill Worker] Starting prefill with TP={tp}")
    
    # Use local model path instead of downloading from HuggingFace
    model_path = "/root/ssd/mxy/models/Llama-3.1-8B-Instruct"
    
    # Load prompts
    if lmeval_task:
        prompts = load_lmeval_prompts(lmeval_task, num_samples=num_requests)
    else:
        # Default prompts for testing
        prompts = [
            "Explain the concept of machine learning in simple terms.",
            "What are the main differences between Python and Java?",
            "Describe the water cycle.",
        ] * (num_requests // 3 + 1)
        prompts = prompts[:num_requests]
    
    # Create simulator
    sim = SimulatorBackend(
        model_path=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.75,
        max_model_len=16384,
        max_batch_size=5,
        dtype="float16",
        network_gbps=80.0,
        max_concurrent_transfers=2,
        network_efficiency=0.9,
        pcie_gbps=32.0,
        compression_config=compression_config,
    )
    
    await sim.initialize()
    
    # Run prefill
    results = await sim.run_prefill_only(
        prompts=prompts,
        max_tokens=100,
        request_rate_rps=request_rate_rps,
        output_file=intermediate_file,
    )
    
    print(f"[INFO] [Prefill Worker] Prefill complete, saved to {intermediate_file}")
    return results


async def worker_decode(
    tp: int,
    intermediate_file: str,
    output_file: str,
    compression_config=None,
):
    """Run decode stage"""
    from kvserve.simulator.simulator_backend import SimulatorBackend
    
    print(f"[INFO] [Decode Worker] Starting decode with TP={tp}")
    
    # Use local model path instead of downloading from HuggingFace
    model_path = "/root/ssd/mxy/models/Llama-3.1-8B-Instruct"
    
    # Create simulator
    sim = SimulatorBackend(
        model_path=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.75,
        max_model_len=16384,
        max_batch_size=5,
        dtype="float16",
        network_gbps=80.0,
        max_concurrent_transfers=2,
        network_efficiency=0.9,
        pcie_gbps=32.0,
        compression_config=compression_config,
    )
    
    await sim.initialize()
    
    # Run decode
    results = await sim.run_decode_only(
        input_file=intermediate_file,
        output_file=output_file,
    )
    
    # Print statistics
    sim.print_stats(results)
    
    print(f"[INFO] [Decode Worker] Decode complete, saved to {output_file}")
    return results


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test Prefill-Decode separation simulator")
    parser.add_argument("config", choices=["tp1", "tp2", "tp4"], help="Configuration (tp1, tp2, tp4)")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of requests to simulate")
    parser.add_argument("--request-rate", type=float, default=1.0, help="Request rate in requests per second")
    parser.add_argument("--lmeval-task", type=str, default=None, help="LM-Eval task name (e.g., longbench_qasper)")
    parser.add_argument("--compress", action="store_true", help="Enable compression")
    
    args = parser.parse_args()
    
    # Parse TP
    tp = int(args.config.replace("tp", ""))
    
    print(f"[INFO] Configuration: TP={tp}, num_requests={args.num_requests}, request_rate={args.request_rate}")
    
    # Compression config
    compression_config = None
    if args.compress:
        compression_config = {
            "enabled": True,
            "pipeline": {
                "transformer": "kvserve",
                "quantizer": "kvserve",
                "codec": "kvserve"
            }
        }
        print(f"[INFO] Compression enabled")
    
    # File paths
    intermediate_file = f"prefill_results_{args.config}.pkl"
    output_file = f"decode_results_{args.config}.pkl"
    
    # Run prefill
    print(f"\n{'='*80}")
    print(f"PREFILL STAGE (TP={tp})")
    print(f"{'='*80}")
    
    try:
        await worker_prefill(
            tp=tp,
            intermediate_file=intermediate_file,
            num_requests=args.num_requests,
            request_rate_rps=args.request_rate,
            lmeval_task=args.lmeval_task,
            compression_config=compression_config,
        )
        print(f"✅ TP={tp} Prefill succeeded")
    except Exception as e:
        print(f"❌ TP={tp} Prefill failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run decode
    print(f"\n{'='*80}")
    print(f"DECODE STAGE (TP={tp})")
    print(f"{'='*80}")
    
    try:
        await worker_decode(
            tp=tp,
            intermediate_file=intermediate_file,
            output_file=output_file,
            compression_config=compression_config,
        )
        print(f"✅ TP={tp} Decode succeeded")
    except Exception as e:
        print(f"❌ TP={tp} Decode failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*80}")
    print(f"ALL STAGES COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())

