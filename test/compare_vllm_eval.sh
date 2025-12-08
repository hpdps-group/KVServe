#!/bin/bash
# Compare KVServe evaluation with native vLLM evaluation
# This script runs the same evaluation using lm_eval's native vLLM backend

MODEL_PATH="/root/ssd/Llama3.1-8B-Instruct"
TASK="hellaswag"
LIMIT=5
NUM_FEWSHOT=0

echo "============================================================"
echo "Running evaluation with native vLLM backend (lm_eval)"
echo "============================================================"
echo "Model: $MODEL_PATH"
echo "Task: $TASK"
echo "Limit: $LIMIT examples"
echo "Num fewshot: $NUM_FEWSHOT"
echo ""

cd /root/lzd/lm-evaluation-harness

# Use v0 engine to avoid CUDA fork issues
# v1 engine uses multiprocessing with 'fork' which doesn't work with CUDA in Python 3.12
# v0 engine uses Ray which handles CUDA properly
export PYTHONUNBUFFERED=1

# Force vLLM to use v0 engine (Ray-based) instead of v1 (multiprocessing-based)
# This avoids "Cannot re-initialize CUDA in forked subprocess" error
export VLLM_USE_V1=0

# Alternative: Set multiprocessing start method to 'spawn' in Python
# But using v0 engine is simpler and more compatible
python3 -c "
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
" 2>/dev/null || true

lm_eval --model vllm \
    --model_args pretrained=$MODEL_PATH,dtype=float16,gpu_memory_utilization=0.85,enforce_eager=True \
    --tasks $TASK \
    --num_fewshot $NUM_FEWSHOT \
    --limit $LIMIT \
    --batch_size auto \
    --verbosity INFO

echo ""
echo "============================================================"
echo "Evaluation completed"
echo "============================================================"
echo ""
echo "To compare with KVServe, run:"
echo "  cd /root/lzd/kvserve_project && python3 test/test_evaluation.py"

