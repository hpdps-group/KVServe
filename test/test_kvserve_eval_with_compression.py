#!/usr/bin/env python3
"""
KVServe Evaluation Test with Compression
Run lm-evaluation-harness style tasks with KV compression enabled
"""

import asyncio
import os
import sys

# Note: If you encounter SSL/connection issues with huggingface.co, you can uncomment below to use mirror
# However, mirror sites may have rate limits. Use direct connection if possible.
# os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ray
from kvserve.engine.backend import PDBackend
from kvserve.eval.runner import run_evaluation

# Model path (update this if needed)
MODEL_PATH = "/root/workspace/models/Llama-3.1-8B-Instruct"

# Compression configuration (based on test_kvserve_manager.py)
COMPRESSION_CONFIG = {
    "enabled": True,
    "pipeline": ["transformer", "quantizer", "codec"],
    "transformer_config": {
        "transform_type": "hadamard",
        "seed": 0x3333,
    },
    "quantizer_config": {
        "model_name": "Llama-3.1-8B-Instruct",
        "hybrid_ratio": 0.8,
        "high_key_max_value": 12,
        "high_value_max_value": 8,
        "low_key_max_value": 6,
        "low_value_max_value": 4,
        "axis_key": "channel",
        "axis_value": "token",
        "split_type": "head",
    },
    "codec_config": {
        "codec_type": "nvcomp",
        "nvcomp_algorithm": "ANS",
        "data_type": "|u1",
    },
    "min_compress_size": 1024,
}


async def run_eval_with_compression():
    """Run evaluation with compression enabled"""
    print("="*70)
    print("KVServe Evaluation Test with KV Compression")
    print("="*70)
    print()
    
    # Initialize Ray
    if not ray.is_initialized():
        print("[1] Initializing Ray...")
        env_pythonpath = os.environ.get("PYTHONPATH", "")
        if project_root not in env_pythonpath:
            os.environ["PYTHONPATH"] = project_root + (":" + env_pythonpath if env_pythonpath else "")
        
        # Pin GPUs to 6 and 7
        # os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6,7")
        
        ray.init(
            ignore_reinit_error=True,
            num_gpus=2,  # 1 prefill + 1 decode
            runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}}
        )
        print("✓ Ray initialized")
        print()
    
    # Create backend with compression
    print("[2] Creating PDBackend with compression...")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Compression: {'ENABLED' if COMPRESSION_CONFIG['enabled'] else 'DISABLED'}")
    print(f"  Pipeline: {' → '.join(COMPRESSION_CONFIG['pipeline'])}")
    print()
    
    backend = PDBackend(
        model_path=MODEL_PATH,
        num_prefill_workers=1,
        num_decoding_workers=1,
        block_size=16,
        dtype="bfloat16",
        gpu_memory_utilization=0.75, 
        kv_transfer_method="nccl",
        nccl_init_method="tcp://localhost:29500",
        log_level="WARNING",
        max_model_len=32768,
        max_batch_size=1,
        compression_config=COMPRESSION_CONFIG,
    )
    
    # Set sampling parameters
    backend.default_temperature = 0.0
    backend.default_top_p = 1.0
    backend.default_top_k = 0.0
    backend.max_new_tokens = 128
    
    print("✓ Backend created")
    print()
    
    try:
        # Initialize and start backend
        print("[3] Initializing backend (loading models)...")
        print("    This may take a few minutes...")
        await backend.initialize()
        await backend.start()
        await asyncio.sleep(2)  # Wait for stability
        print("✓ Backend initialized and started")
        print()
        
        # Run evaluation (report real params instead of hardcoded text)
        eval_tasks = ["longbench_qasper"]
        eval_batch_size = 1
        eval_limit = None  # limit disabled by default; set an int to cap examples

        print("[4] Running evaluation...")
        print(f"    Tasks: {eval_tasks}")
        print(f"    Batch size: {eval_batch_size}")
        print(f"    Limit: {eval_limit if eval_limit is not None else 'not set'}")
        print()
        
        results = run_evaluation(
            backend=backend,
            tasks=eval_tasks,
            model_path=MODEL_PATH,
            num_fewshot=None,
            batch_size=eval_batch_size,
            #limit=eval_limit,
            verbosity="WARNING",
            output_path=None,
            apply_chat_template=True,
        )
        
        # Print results
        print("\n" + "="*70)
        print("Evaluation Results")
        print("="*70)
        if "results" in results:
            for task_name, task_results in results["results"].items():
                print(f"\n{task_name}:")
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
        print("="*70)
        print()
        
        return results
        
    finally:
        print("[5] Stopping backend...")
        try:
            await backend.stop()
            print("✓ Backend stopped")
        except Exception as e:
            print(f"⚠️  Backend stop warning: {e}")
        print()


if __name__ == "__main__":
    try:
        results = asyncio.run(run_eval_with_compression())
        print("✓ Evaluation completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n✗ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

