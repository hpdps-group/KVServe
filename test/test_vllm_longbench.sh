#!/bin/bash
# Test longbench_2wikimqa with native vLLM backend

# Force vLLM to use v0 engine (Ray-based) to avoid CUDA fork issues
export VLLM_USE_V1=0
export VLLM_ENGINE_V1=0

cd /root/lzd/lm-evaluation-harness

# Add enforce_eager to disable CUDA graphs (safer for evaluation)
lm_eval --model vllm \
    --model_args pretrained=/root/ssd/Llama3.1-8B-Instruct,max_batch_size=2,dtype=float16,tensor_parallel_size=1,gpu_memory_utilization=0.85,max_model_len=20000,enforce_eager=True \
    --tasks longbench_2wikimqa \
    --apply_chat_template True
    # --num_fewshot 0 \
    # --limit 5 \


