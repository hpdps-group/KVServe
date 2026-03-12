#!/usr/bin/env python3
"""
KVServe TP (Tensor Parallel) Test
Test PD separation with Tensor Parallelism (no compression first)
"""

import asyncio
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ray
from transformers import AutoTokenizer
from kvserve.engine.backend import PDBackend
from kvserve.engine.utils import Request

# Model path
MODEL_PATH = "/root/ssd/Llama3.1-8B-Instruct"

# ==============================================================================
# TP Configuration
# ==============================================================================
# TP = Tensor Parallel Size (number of GPUs for model sharding)
# - TP=1: No model sharding, 1 worker per stage, 2 GPUs total
# - TP=2: Model sharded across 2 GPUs, 2 workers per stage, 4 GPUs total
#
# Architecture (TP=2 example):
#   Prefill Stage: 2 workers (TP ranks 0,1) on GPUs 0,1
#   Decode Stage:  2 workers (TP ranks 0,1) on GPUs 2,3
#   P2P Transfer:  Prefill[i] → Decode[i] (shard-to-shard)
TENSOR_PARALLEL_SIZE = 2  # Set to 2 for TP=2 testing
NUM_PREFILL_WORKERS = 1   # Not used when TP>1 (will create TP_SIZE workers)
NUM_DECODING_WORKERS = 1  # Not used when TP>1 (will create TP_SIZE workers)


async def run_tp_test():
    """Run TP test without compression"""
    print("="*70)
    print("KVServe Tensor Parallel (TP) Test")
    print("="*70)
    print()
    
    # Initialize Ray
    if not ray.is_initialized():
        print("[1] Initializing Ray...")
        env_pythonpath = os.environ.get("PYTHONPATH", "")
        if project_root not in env_pythonpath:
            os.environ["PYTHONPATH"] = project_root + (":" + env_pythonpath if env_pythonpath else "")
        
        # Calculate total GPUs needed
        total_gpus = (NUM_PREFILL_WORKERS + NUM_DECODING_WORKERS) * TENSOR_PARALLEL_SIZE
        
        # Pin GPUs based on TP size
        if TENSOR_PARALLEL_SIZE == 1:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6,7")  # 2 GPUs
        elif TENSOR_PARALLEL_SIZE == 2:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6,7,4,5")  # 4 GPUs
        
        ray.init(
            ignore_reinit_error=True,
            num_gpus=total_gpus,
            runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}}
        )
        print(f"✓ Ray initialized with {total_gpus} GPUs (TP={TENSOR_PARALLEL_SIZE})")
        print()
    
    # Create backend WITHOUT compression
    print("[2] Creating PDBackend...")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
    print(f"  Compression: DISABLED")
    print()
    
    backend = PDBackend(
        model_path=MODEL_PATH,
        num_prefill_workers=NUM_PREFILL_WORKERS,
        num_decoding_workers=NUM_DECODING_WORKERS,
        block_size=16,
        dtype="float16",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=0.75,
        kv_transfer_method="nccl",
        nccl_init_method="tcp://localhost:29500",
        log_level="DEBUG",  # Changed to DEBUG for more detailed logs
        max_model_len=2048,
        max_batch_size=4,
        compression_config=None,  # No compression
    )
    
    # Set sampling parameters
    backend.default_temperature = 0.0
    backend.default_top_p = 1.0
    backend.default_top_k = 0.0
    backend.max_new_tokens = 50
    
    print("✓ Backend created")
    print()
    
    # Load tokenizer
    print("[3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("✓ Tokenizer loaded")
    print()
    
    try:
        # Initialize and start backend
        print("[4] Initializing backend (loading models)...")
        await backend.initialize()
        await backend.start()
        await asyncio.sleep(2)
        print("✓ Backend initialized and started")
        print()
        
        # Test prompts
        print("[5] Running test generation...")
        test_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about nature.",
        ]
        
        # Submit all requests first
        requests = []
        for i, prompt in enumerate(test_prompts):
            print(f"  Preparing request {i+1}/{len(test_prompts)}: {prompt[:50]}...")
            
            # Tokenize prompt
            prompt_token_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
            
            request = Request(
                request_id=f"test_{i}",
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                max_tokens=50,
                temperature=0.0,
            )
            requests.append(request)
            await backend.add_request(request)
        
        print(f"\n✓ All {len(requests)} requests submitted")
        
        # Collect outputs by polling
        completed = set()
        all_outputs = {}
        max_iterations = 500
        iteration = 0
        last_progress_print = 0
        
        print("\nGenerating outputs...")
        print(f"  Waiting for {len(requests)} requests to complete...")
        
        while len(completed) < len(requests) and iteration < max_iterations:
            iteration += 1
            new_outputs = await backend.get_outputs()
            
            # Print progress every 50 iterations
            if iteration - last_progress_print >= 50:
                print(f"  [Iteration {iteration}] Outputs received: {len(all_outputs)}, Completed: {len(completed)}/{len(requests)}")
                # Check scheduler status
                if hasattr(backend, 'prefill_engine') and backend.prefill_engine:
                    p_waiting = backend.prefill_engine.scheduler.num_waiting_requests()
                    p_running = backend.prefill_engine.scheduler.num_running_requests()
                    print(f"    Prefill: waiting={p_waiting}, running={p_running}")
                if hasattr(backend, 'decode_engine') and backend.decode_engine:
                    d_waiting = backend.decode_engine.scheduler.num_waiting_requests()
                    d_running = backend.decode_engine.scheduler.num_running_requests()
                    print(f"    Decode: waiting={d_waiting}, running={d_running}")
                last_progress_print = iteration
            
            for output in new_outputs:
                request_id = output.request_id
                if request_id not in all_outputs:
                    all_outputs[request_id] = []
                
                # Only append if this is a new output
                if not all_outputs[request_id] or len(output.output_token_ids) > len(all_outputs[request_id][-1].output_token_ids):
                    all_outputs[request_id].append(output)
                    
                    if output.finished:
                        completed.add(request_id)
                        decoded_text = tokenizer.decode(output.output_token_ids)
                        print(f"  ✓ [{request_id}] Completed: {decoded_text}")
            
            await asyncio.sleep(0.1)
        
        # Print final status
        if iteration >= max_iterations:
            print(f"\n⚠️  Timeout after {max_iterations} iterations")
            print(f"  Outputs received: {len(all_outputs)}, Completed: {len(completed)}/{len(requests)}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("Summary:")
        print(f"{'='*70}")
        for i, request in enumerate(requests):
            request_id = request.request_id
            if request_id in all_outputs and all_outputs[request_id]:
                final_output = all_outputs[request_id][-1]
                decoded_text = tokenizer.decode(final_output.output_token_ids)
                print(f"\nRequest {i+1}: {request.prompt}")
                print(f"  Tokens: {len(final_output.output_token_ids)}")
                print(f"  Output: {decoded_text}")
            else:
                print(f"\nRequest {i+1}: {request.prompt}")
                print(f"  NO OUTPUT RECEIVED")
        
        print("\n" + "="*70)
        print("✓ TP test completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n[6] Stopping backend...")
        try:
            await asyncio.wait_for(backend.stop(), timeout=30.0)
            print("✓ Backend stopped")
        except asyncio.TimeoutError:
            print("⚠️  Backend stop timed out after 30s")
        except Exception as e:
            print(f"⚠️  Error stopping backend: {e}")


if __name__ == "__main__":
    asyncio.run(run_tp_test())

