#!/usr/bin/env python3
"""
Test Worker GPU allocation
"""

import os
import sys
import ray
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Add vLLM path
vllm_path = "/root/lzd/vllm-0.10.1"
sys.path.insert(0, vllm_path)

from kvserve.engine.utils import EngineStage
from kvserve.engine.worker import Worker

def main():
    """Test worker GPU allocation"""
    print("\n" + "=" * 60)
    print("🧪 Testing Worker GPU Allocation")
    print("=" * 60)
    print()
    
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    
    ray.init(
        num_gpus=2,
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": f"{vllm_path}:{project_root}"}}
    )
    
    print("✅ Ray initialized with 2 GPUs")
    print()
    
    try:
        # Create two workers
        print("Creating Worker 1 (Prefill, should get GPU 0)...")
        worker1 = Worker.remote(
            worker_id=0,
            stage=EngineStage.PREFILL,
            model_path="/root/lzd/model/qwen2.5-VL",
            global_rank=0,
            world_size=2,
            nccl_init_method="tcp://localhost:29500",
        )
        
        print("Creating Worker 2 (Decode, should get GPU 1)...")
        worker2 = Worker.remote(
            worker_id=0,
            stage=EngineStage.DECODING,
            model_path="/root/lzd/model/qwen2.5-VL",
            global_rank=1,
            world_size=2,
            nccl_init_method="tcp://localhost:29500",
        )
        
        # Wait for workers to be ready
        print("\nWaiting for workers to initialize...")
        ray.get([
            worker1.ready.remote(),
            worker2.ready.remote()
        ])
        
        print("\n✅ Both workers initialized successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Worker initialization failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
    
    finally:
        ray.shutdown()
        print("\n🛑 Ray shutdown complete")

if __name__ == "__main__":
    main()


