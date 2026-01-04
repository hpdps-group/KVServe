#!/usr/bin/env python3
"""
GPU Environment Diagnostic Script
"""

import os
import sys
import torch
import ray

def check_torch_cuda():
    """Check PyTorch CUDA configuration"""
    print("=" * 60)
    print("🔍 PyTorch CUDA Configuration")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}:")
            print(f"  Name: {props.name}")
            print(f"  Total memory: {props.total_memory / 1024**3:.2f} GiB")
            print(f"  Compute capability: {props.major}.{props.minor}")
    else:
        print("❌ CUDA not available!")
    print()

def check_cuda_visible_devices():
    """Check CUDA_VISIBLE_DEVICES environment variable"""
    print("=" * 60)
    print("🔍 Environment Variables")
    print("=" * 60)
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print()

def check_ray_gpu():
    """Check Ray GPU allocation"""
    print("=" * 60)
    print("🔍 Ray GPU Configuration")
    print("=" * 60)
    
    try:
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_gpus=2, ignore_reinit_error=True)
        
        @ray.remote(num_gpus=1)
        class TestActor:
            def get_gpu_info(self):
                import os
                import torch
                gpu_ids = ray.get_gpu_ids()
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
                return {
                    "ray_gpu_ids": gpu_ids,
                    "cuda_visible_devices": cuda_visible,
                    "torch_device_count": torch.cuda.device_count(),
                }
        
        # Test with 2 actors
        actor1 = TestActor.remote()
        actor2 = TestActor.remote()
        
        info1 = ray.get(actor1.get_gpu_info.remote())
        info2 = ray.get(actor2.get_gpu_info.remote())
        
        print("Actor 1:")
        for key, value in info1.items():
            print(f"  {key}: {value}")
        
        print("\nActor 2:")
        for key, value in info2.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Ray GPU allocation successful!")
        
    except Exception as e:
        print(f"❌ Ray GPU allocation failed: {e}")
    finally:
        if ray.is_initialized():
            ray.shutdown()
    print()

def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("🔧 GPU Environment Diagnostic Tool")
    print("=" * 60)
    print()
    
    check_torch_cuda()
    check_cuda_visible_devices()
    check_ray_gpu()
    
    print("=" * 60)
    print("✅ Diagnostic complete")
    print("=" * 60)

if __name__ == "__main__":
    main()


