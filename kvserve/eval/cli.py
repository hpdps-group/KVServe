#!/usr/bin/env python3
"""
Command-line interface for KVServe evaluation
Similar to lm_eval, but uses KVServe backend
"""

import argparse
import asyncio
import os
import sys
import json
from typing import List, Optional, Union, Dict, Any

# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ray
from kvserve.engine.backend import PDBackend
from kvserve.eval.runner import run_evaluation


def parse_model_args(model_args_str: str) -> Dict[str, Any]:
    """
    Parse model arguments string (e.g., "pretrained=/path/to/model,num_prefill_workers=2")
    
    Args:
        model_args_str: Comma-separated key=value pairs
        
    Returns:
        Dictionary of parsed arguments
    """
    if not model_args_str:
        return {}
    
    args = {}
    for pair in model_args_str.split(','):
        pair = pair.strip()
        if '=' not in pair:
            continue
        
        key, value = pair.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Try to convert to appropriate type
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string
        
        args[key] = value
    
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models using KVServe backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  kvserve_eval --model kvserve --model_args pretrained=/path/to/model --tasks hellaswag
  
  # With engine parameters
  kvserve_eval --model kvserve \\
    --model_args pretrained=/path/to/model,num_prefill_workers=2,num_decoding_workers=2 \\
    --tasks hellaswag mmlu \\
    --num_fewshot 0 \\
    --limit 100
  
  # Save results to file
  kvserve_eval --model kvserve \\
    --model_args pretrained=/path/to/model \\
    --tasks hellaswag \\
    --output_path results.json
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="kvserve",
        help="Model backend (currently only 'kvserve' is supported)"
    )
    
    # Model arguments (similar to lm_eval)
    parser.add_argument(
        "--model_args",
        type=str,
        default="",
        help="Comma-separated model arguments (e.g., pretrained=/path/to/model,num_prefill_workers=2)"
    )
    
    # Tasks
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help="Tasks to evaluate (e.g., hellaswag mmlu)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples"
    )
    
    parser.add_argument(
        "--limit",
        type=str,
        default=None,
        help="Limit number of examples (int) or fraction (float)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    
    # Output
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save results JSON"
    )
    
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        choices=["ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity level"
    )
    
    # Ray configuration
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs for Ray (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    # Parse limit value
    if args.limit:
        if '.' in args.limit:
            args.limit = float(args.limit)
        else:
            args.limit = int(args.limit)
    
    # Parse model arguments
    model_args = parse_model_args(args.model_args)
    
    # Extract required arguments
    model_path = model_args.pop("pretrained", None)
    if not model_path:
        parser.error("--model_args must include 'pretrained=/path/to/model'")
    
    # Extract KVServe-specific arguments
    num_prefill_workers = model_args.pop("num_prefill_workers", 1)
    num_decoding_workers = model_args.pop("num_decoding_workers", 1)
    block_size = model_args.pop("block_size", 16)
    max_num_gpu_blocks = model_args.pop("max_num_gpu_blocks", 3000)
    dtype = model_args.pop("dtype", "float16")
    gpu_memory_utilization = model_args.pop("gpu_memory_utilization", 0.85)
    kv_transfer_method = model_args.pop("kv_transfer_method", "nccl")
    nccl_init_method = model_args.pop("nccl_init_method", "tcp://localhost:29500")
    log_level = model_args.pop("log_level", "WARNING")
    max_model_len = model_args.pop("max_model_len", 32768)
    max_batch_size = model_args.pop("max_batch_size", 32)
    
    # Extract sampling parameters (for evaluation requests)
    default_temperature = model_args.pop("temperature", None)
    default_top_p = model_args.pop("top_p", None)
    default_top_k = model_args.pop("top_k", None)
    
    # Calculate number of GPUs needed
    if args.num_gpus is None:
        num_gpus = num_prefill_workers + num_decoding_workers
    else:
        num_gpus = args.num_gpus
    
    async def run_eval():
        # Initialize Ray
        if not ray.is_initialized():
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env_pythonpath = os.environ.get("PYTHONPATH", "")
            if project_root not in env_pythonpath:
                os.environ["PYTHONPATH"] = project_root + (":" + env_pythonpath if env_pythonpath else "")
            
            ray.init(
                ignore_reinit_error=True,
                num_gpus=num_gpus,
                runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}}
            )
        
        # Create backend
        backend = PDBackend(
            model_path=model_path,
            num_prefill_workers=num_prefill_workers,
            num_decoding_workers=num_decoding_workers,
            block_size=block_size,
            max_num_gpu_blocks=max_num_gpu_blocks,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_transfer_method=kv_transfer_method,
            nccl_init_method=nccl_init_method,
            log_level=log_level,
            max_model_len=max_model_len,
            max_batch_size=max_batch_size,
        )
        
        # Store default sampling parameters for evaluator
        backend.default_temperature = default_temperature
        backend.default_top_p = default_top_p
        backend.default_top_k = default_top_k
        
        try:
            # Initialize and start backend
            await backend.initialize()
            await backend.start()
            await asyncio.sleep(2)  # Wait for stability
            
            # Run evaluation
            results = run_evaluation(
                backend=backend,
                tasks=args.tasks,
                model_path=model_path,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                limit=args.limit,
                verbosity=args.verbosity,
                output_path=args.output_path,
            )
            
            # Print results summary
            if "results" in results:
                print("\n" + "="*60)
                print("Evaluation Results")
                print("="*60)
                for task_name, task_results in results["results"].items():
                    print(f"\n{task_name}:")
                    for metric, value in task_results.items():
                        if isinstance(value, (int, float)):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: {value}")
                print("="*60)
            
            return results
            
        finally:
            await backend.stop()
    
    # Run evaluation
    try:
        results = asyncio.run(run_eval())
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nEvaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

