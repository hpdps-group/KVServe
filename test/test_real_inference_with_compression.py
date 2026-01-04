#!/usr/bin/env python3
"""
Test real inference with KV compression
Runs 20 prompts through Qwen model with compression enabled
"""

import asyncio
import os
import sys
import ray
from transformers import AutoTokenizer

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from kvserve.engine.backend import PDBackend
from kvserve.engine.utils import Request


async def test_real_inference_with_compression():
    """Test real inference with KV compression enabled"""
    
    print("=" * 70)
    print("Real Inference Test with KV Compression")
    print("=" * 70)
    print()
    
    # Initialize Ray
    if not ray.is_initialized():
        print("[1] Initializing Ray...")
        # Set PYTHONPATH so Ray workers can find kvserve module and correct vLLM version
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vllm_path = "/root/lzd/vllm-0.10.1"  # Use vLLM 0.10.1
        env_pythonpath = os.environ.get("PYTHONPATH", "")
        
        # Build PYTHONPATH: vllm-0.10.1 first, then project root
        pythonpath_parts = [vllm_path, project_root]
        if env_pythonpath:
            pythonpath_parts.append(env_pythonpath)
        os.environ["PYTHONPATH"] = ":".join(pythonpath_parts)
        
        print(f"  Using vLLM from: {vllm_path}")
        
        ray.init(
            ignore_reinit_error=True,
            num_gpus=2,  # Need at least 2 GPUs for PD separation
            runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}}
        )
        print("✓ Ray initialized")
        print()
    
    # Model path
    model_path = "/root/lzd/model/qwen2.5-VL"
    if not os.path.exists(model_path):
        print(f"✗ Model not found at {model_path}")
        print("  Please update model_path in the script")
        return
    
    # Compression configuration
    print("[2] Configuring KV compression...")
    # Note: Qwen model may have different num_heads than Llama
    # Using layer split to avoid head count mismatch
    compression_config = {
        "enabled": True,
        "pipeline": ["quantizer"],  # Test quantizer only first
        "transformer_config": {
            "transform_type": "hadamard",
            "seed": 0x3333,
        },
        "quantizer_config": {
            "model_name": "Llama-3.1-8B-Instruct",  # For head scores CSV
            "split_type": "head",  # Use head split
            "hybrid_ratio": 0.5,  # 50% heads use low precision
            "high_key_max_value": 16,
            "low_key_max_value": 12,
            "axis_key": "channel",
            "axis_value": "token",
        },
        "codec_config": {
            "codec_type": "nvcomp",
            "nvcomp_algorithm": "LZ4",
        },
        "min_compress_size": 1024,
    }
    print(f"  ✓ Compression enabled: {compression_config['pipeline']}")
    print(f"  ✓ Pipeline: {' → '.join(compression_config['pipeline'])}")
    print()
    
    # Create backend
    print("[3] Creating PD Backend with compression...")
    backend = PDBackend(
        model_path=model_path,
        num_prefill_workers=1,
        num_decoding_workers=1,
        enable_multi_stream=False,  # Single stream for now
        block_size=16,
        dtype="float16",
        gpu_memory_utilization=0.85,
        kv_transfer_method="nccl",
        nccl_init_method="tcp://localhost:29500",
        compression_config=compression_config,  # Enable compression
        max_model_len=4500,
        max_batch_size=10,
    )
    print("✓ Backend created")
    print()
    
    # Initialize backend
    print("[4] Initializing backend (loading models)...")
    print("    This may take a few minutes...")
    await backend.initialize()
    print("✓ Backend initialized")
    print()
    
    # Start backend
    print("[5] Starting backend...")
    await backend.start()
    await asyncio.sleep(2)  # Wait for stability
    print("✓ Backend started")
    print()
    
    # Load tokenizer for creating test requests
    print("[6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("✓ Tokenizer loaded")
    print()
    
    # Create 20 test prompts
    print("[7] Creating 20 test prompts...")
    test_prompts = [
        "Explain the theory of quantum mechanics and its applications.",
        "Describe the process of photosynthesis in plants in detail.",
        "What is artificial intelligence and how does it work?",
        "Tell me about the history of the internet and its development.",
        "How do neural networks learn from data?",
        "Explain the differences between machine learning and deep learning.",
        "What is the structure of DNA and how does it store genetic information?",
        "Describe the water cycle and its importance to Earth's climate.",
        "How do computers process and store information?",
        "What are the main causes of climate change and their effects?",
        "Explain the concept of evolution by natural selection.",
        "What is the difference between renewable and non-renewable energy?",
        "How do vaccines work to protect against diseases?",
        "Describe the process of cellular respiration.",
        "What is the theory of relativity and why is it important?",
        "Explain how the human brain processes information.",
        "What are the key principles of sustainable development?",
        "How do search engines index and retrieve web pages?",
        "Describe the structure and function of the solar system.",
        "What is blockchain technology and how does it work?",
    ]
    
    requests = []
    for i, prompt in enumerate(test_prompts, 1):
        request_id = f"req_{i:03d}"
        prompt_token_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        
        request = Request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_tokens=50,
            temperature=0.8,
            top_p=0.9,
        )
        requests.append(request)
        print(f"  [{i:2d}] {prompt[:60]}... ({len(prompt_token_ids)} tokens)")
    
    print(f"\n✓ Created {len(requests)} test requests")
    print()
    
    # Add all requests to backend
    print("[8] Submitting requests to backend...")
    for request in requests:
        await backend.add_request(request)
    print(f"✓ Submitted {len(requests)} requests")
    print()
    
    # Collect outputs for all requests
    print("[9] Collecting outputs (this will run inference with compression)...")
    print()
    
    completed_requests = set()
    all_outputs = []
    start_time = asyncio.get_event_loop().time()
    
    while len(completed_requests) < len(requests):
        # Check for new outputs
        while backend.output_queue:
            output = backend.output_queue.popleft()
            
            if output.request_id not in completed_requests:
                all_outputs.append(output)
                
                if output.finished:
                    completed_requests.add(output.request_id)
                    request_id = output.request_id
                    output_text = tokenizer.decode(output.output_token_ids, skip_special_tokens=True)
                    
                    # Find the original prompt
                    orig_prompt = next((r.prompt for r in requests if r.request_id == request_id), "Unknown")
                    
                    print(f"[{len(completed_requests):2d}/{len(requests)}] {request_id}")
                    print(f"  Prompt: {orig_prompt[:70]}...")
                    print(f"  Output: {output_text[:100]}...")
                    print()
        
        # Small sleep to avoid busy waiting
        await asyncio.sleep(0.1)
    
    elapsed_time = asyncio.get_event_loop().time() - start_time
    
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Total requests: {len(requests)}")
    print(f"Completed: {len(completed_requests)}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/second")
    print()
    
    # Show sample outputs
    print("Sample outputs:")
    print("-" * 70)
    for i, output in enumerate(all_outputs[:5], 1):  # Show first 5
        if output.finished:
            request_id = output.request_id
            output_text = tokenizer.decode(output.output_token_ids, skip_special_tokens=True)
            orig_prompt = next((r.prompt for r in requests if r.request_id == request_id), "Unknown")
            
            print(f"\n[{i}] Request: {request_id}")
            print(f"Prompt: {orig_prompt}")
            print(f"Output: {output_text}")
    
    print()
    print("=" * 70)
    print("✓ Test completed successfully!")
    print("=" * 70)
    
    # Cleanup
    await backend.stop()
    ray.shutdown()


if __name__ == "__main__":
    asyncio.run(test_real_inference_with_compression())

