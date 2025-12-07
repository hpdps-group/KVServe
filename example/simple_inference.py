#!/usr/bin/env python3
"""
Simple inference example for kvserve
Demonstrates basic usage of PD separation for text generation
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


async def main():
    """Simple inference example"""
    
    # Initialize Ray
    if not ray.is_initialized():
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_pythonpath = os.environ.get("PYTHONPATH", "")
        if project_root not in env_pythonpath:
            os.environ["PYTHONPATH"] = project_root + (":" + env_pythonpath if env_pythonpath else "")
        
        ray.init(
            ignore_reinit_error=True,
            num_gpus=2,  # Need at least 2 GPUs
            runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}}
        )
    
    # Model path
    model_path = "/root/ssd/Llama3.1-8B-Instruct"  # Update this to your model path
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please update model_path in the script")
        return
    
    # Create and initialize backend
    # Set log_level to "WARNING" to suppress debug output, or "DEBUG" to see all logs
    backend = PDBackend(
        model_path=model_path,
        num_prefill_workers=1,
        num_decoding_workers=1,
        block_size=16,
        max_num_gpu_blocks=3000,
        dtype="float16",
        gpu_memory_utilization=0.85,
        kv_transfer_method="nccl",
        nccl_init_method="tcp://localhost:29500",
        log_level="WARNING",  # Options: "ERROR", "WARNING", "INFO", "DEBUG"
    )
    
    await backend.initialize()
    await backend.start()
    await asyncio.sleep(2)  # Wait for stability
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Create a simple request
    prompt = "Explain quantum computing in simple terms."
    prompt_token_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
    
    request = Request(
        request_id="simple_req_001",
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        max_tokens=100,
        temperature=0.8,
        top_p=0.9,
    )
    
    # Submit request
    await backend.add_request(request)
    
    # Collect output
    completed = False
    final_output = None
    
    while not completed:
        outputs = await backend.get_outputs()
        for output in outputs:
            if output.request_id == request.request_id:
                final_output = output
                if output.finished:
                    completed = True
                    break
        
        if not completed:
            await asyncio.sleep(0.1)
    
    # Print result
    if final_output:
        generated_text = tokenizer.decode(final_output.output_token_ids, skip_special_tokens=True)
        total_tokens = len(final_output.output_token_ids)
        print(f"✓ Inference successful. Total tokens: {total_tokens}")
    
    # Cleanup
    await backend.stop()


if __name__ == "__main__":
    asyncio.run(main())

