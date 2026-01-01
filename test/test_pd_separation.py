#!/usr/bin/env python3
"""
Test script for PD separation
Tests Prefill-Decode separation with KV cache transfer
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


async def test_pd_separation():
    """Test PD separation with a simple request"""
    
    # Initialize Ray
    if not ray.is_initialized():
        print("\n[1] Initializing Ray...")
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
            num_gpus=2,  # Need at least 2 GPUs
            runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}}
        )
        print("✓ Ray initialized")
    
    # Model path
    model_path = "/root/lzd/model/qwen2.5-VL"  # Update this to your model path
    if not os.path.exists(model_path):
        print(f"\n✗ Model not found at {model_path}")
        print("  Please update model_path in the script")
        return
    
    # Create backend
    print("\n[2] Creating PD Backend...")
    backend = PDBackend(
        model_path=model_path,
        num_prefill_workers=1,
        num_decoding_workers=1,
        enable_multi_stream=True,
        block_size=16,
        # max_num_gpu_blocks: Auto-profiled ✅
        # max_num_cpu_blocks: Auto-profiled ✅
        dtype="float16",
        gpu_memory_utilization=0.85,
        kv_transfer_method="nccl",
        nccl_init_method="tcp://localhost:29500",
    )
    print("✓ Backend created")
    
    # Initialize backend
    print("\n[3] Initializing backend (loading models)...")
    print("    This may take a few minutes...")
    await backend.initialize()
    print("✓ Backend initialized")
    
    # Start backend
    print("\n[4] Starting backend...")
    await backend.start()
    await asyncio.sleep(2)  # Wait for stability
    print("✓ Backend started")
    
    # Load tokenizer for creating test requests
    print("\n[5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("✓ Tokenizer loaded")
    
    # Create test requests with multiple prompts
    print("\n[6] Creating test requests...")
    
    # Create prompts with varying lengths to test different KV cache sizes
    # Short prompts (1-2 blocks)
    short_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]
    
    # Medium prompts (2-4 blocks) - Repeat to increase length
    medium_prompts = [
        "Write a detailed explanation of quantum computing including its basic principles, applications, and potential impact on technology. " * 2,
        "Describe the process of photosynthesis in detail, explaining how plants convert light energy into chemical energy, and the role of chlorophyll and other molecules. " * 2,
        "Explain how machine learning algorithms work, including supervised learning, unsupervised learning, and reinforcement learning approaches. " * 2,
        "Tell me about the history of the internet, from its origins as ARPANET to the modern world wide web and cloud computing era. " * 2,
        "What is artificial intelligence and how does it differ from machine learning? Explain the various types of AI systems and their applications. " * 2,
    ]
    
    # Long prompts (5-8 blocks) - Much longer to test larger KV cache transfers
    long_prompts = [
        "Write a comprehensive essay about renewable energy sources including solar power, wind energy, hydroelectric power, and geothermal energy. Explain the benefits of each, their environmental impact, technological challenges, cost considerations, and future potential. Discuss how these technologies can help reduce carbon emissions and combat climate change. " * 4,
        "Describe in detail the structure and function of biological cells. Explain the differences between plant and animal cells, the role of various organelles like mitochondria, nucleus, endoplasmic reticulum, and Golgi apparatus. Discuss how cells communicate, divide, and maintain homeostasis. Include information about cellular respiration, protein synthesis, and cell membrane transport mechanisms. " * 4,
        "Explain the theory of relativity comprehensively, covering both special and general relativity. Discuss Einstein's postulates, time dilation, length contraction, mass-energy equivalence, gravitational effects, and their implications for our understanding of space, time, and the universe. " * 4,
        "What are the main causes of climate change and how do they affect our planet? Discuss greenhouse gases, deforestation, industrial activities, transportation emissions, and agricultural practices. Explain the scientific consensus on climate change, observed impacts like rising sea levels and extreme weather, and potential mitigation strategies. " * 4,
        "Describe the process of evolution by natural selection in detail. Explain how genetic variation, inheritance, selection pressures, and time lead to species adaptation and diversification. Discuss evidence for evolution from fossils, comparative anatomy, molecular biology, and biogeography. Include examples of evolutionary mechanisms like mutation, genetic drift, and gene flow. " * 3,
    ]
    
    # Mix of all lengths
    test_prompts = short_prompts + medium_prompts + long_prompts
    
    # Ensure we have at least 20 prompts
    while len(test_prompts) < 20:
        test_prompts.extend(medium_prompts[:2])  # Add more medium prompts if needed
    
    requests = []
    for i, prompt in enumerate(test_prompts, 1):
        request_id = f"test_req_{i:03d}"
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
        print(f"  Created request {request_id}: {prompt[:50]}... ({len(prompt_token_ids)} tokens)")
    
    print(f"\n✓ Created {len(requests)} test requests")
    
    # Add all requests to backend
    print("\n[7] Submitting requests to backend...")
    for request in requests:
        await backend.add_request(request)
    print(f"✓ Submitted {len(requests)} requests")
    
    # Collect outputs for all requests
    print("\n[8] Collecting outputs...")
    completed_requests = set()
    all_outputs = {}  # request_id -> list of outputs
    
    max_iterations = 1000  # Prevent infinite loop
    iteration = 0
    
    while len(completed_requests) < len(requests) and iteration < max_iterations:
        iteration += 1
        new_outputs = await backend.get_outputs()
        
        for output in new_outputs:
            request_id = output.request_id
            if request_id not in all_outputs:
                all_outputs[request_id] = []
            
            # Only append if this is a new output (check token count)
            if not all_outputs[request_id] or len(output.output_token_ids) > len(all_outputs[request_id][-1].output_token_ids):
                all_outputs[request_id].append(output)
                
                tokens = tokenizer.decode(output.output_token_ids)
                print(f"  [{request_id}] Step: {len(output.output_token_ids)} tokens - {tokens[:60]}...")
                
                if output.finished:
                    completed_requests.add(request_id)
                    print(f"  ✓ [{request_id}] Completed: {len(output.output_token_ids)} tokens total")
        
        await asyncio.sleep(0.1)
        
        # Print progress periodically
        if iteration % 50 == 0:
            print(f"  Progress: {len(completed_requests)}/{len(requests)} requests completed")
    
    # Print summary for all requests
    print("\n[9] Summary of all requests:")
    for i, request in enumerate(requests, 1):
        request_id = request.request_id
        if request_id in all_outputs and all_outputs[request_id]:
            final_output = all_outputs[request_id][-1]
            decoded_text = tokenizer.decode(final_output.output_token_ids)
            print(f"\n  Request {i:2d} ({request_id}):")
            print(f"    Prompt: {request.prompt[:60]}...")
            print(f"    Tokens: {len(final_output.output_token_ids)}")
            print(f"    Output: {decoded_text[:100]}...")
        else:
            print(f"\n  Request {i:2d} ({request_id}): NO OUTPUT RECEIVED")
    
    # Print statistics
    print("\n[10] Statistics:")
    stats = backend.get_stats()
    print(f"  Prefill: {stats.get('prefill', {})}")
    print(f"  Decoding: {stats.get('decoding', {})}")
    print(f"  KV Transfer: {stats.get('kv_transfer', {})}")
    print(f"  Completed requests: {len(completed_requests)}/{len(requests)}")
    
    # Stop backend
    print("\n[11] Stopping backend...")
    await backend.stop()
    print("✓ Backend stopped")
    
    print("\n✅ PD separation test completed!")


if __name__ == "__main__":
    asyncio.run(test_pd_separation())

