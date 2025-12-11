#!/bin/bash
# KVServe Evaluation Test Script
# Run lm-evaluation-harness style tasks with KVServe backend

# Change to project directory
cd "$(dirname "$0")/.."

# Run evaluation
python -m kvserve.eval.cli \
    --model kvserve \
    --model_args "pretrained=/root/ssd/Llama3.1-8B-Instruct, \
    num_prefill_workers=1,num_decoding_workers=1,max_model_len=8000, \
    max_new_tokens=512,apply_chat_template=False" \
    --tasks gsm8k_cot \
    --batch_size 16 \
    --limit 200

echo ""
echo "Evaluation completed!"

