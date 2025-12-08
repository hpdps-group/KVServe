# KVServe Evaluation Module

Evaluation module for KVServe, compatible with lm-evaluation-harness interface.

## Overview

This module provides evaluation capabilities for KVServe backend, allowing you to evaluate models on standard benchmarks using the same interface as lm-evaluation-harness.

## Installation

Install lm-evaluation-harness:

```bash
pip install lm-eval
```

## Usage

### Basic Usage

```python
import asyncio
import ray
from kvserve.engine.backend import PDBackend
from kvserve.eval import run_evaluation

async def main():
    # Initialize Ray
    ray.init(num_gpus=2)
    
    # Create and initialize backend
    backend = PDBackend(
        model_path="/path/to/model",
        num_prefill_workers=1,
        num_decoding_workers=1,
        log_level="WARNING",
    )
    await backend.initialize()
    await backend.start()
    
    # Run evaluation
    results = run_evaluation(
        backend=backend,
        tasks=["hellaswag", "mmlu"],
        model_path="/path/to/model",
        num_fewshot=0,
        limit=100,  # Evaluate on first 100 examples
    )
    
    print(results)
    
    await backend.stop()

asyncio.run(main())
```

### Command Line Interface

```bash
python -m kvserve.eval.runner \
    --model_path /path/to/model \
    --tasks hellaswag mmlu \
    --num_fewshot 0 \
    --limit 100 \
    --output_path results.json \
    --num_prefill_workers 1 \
    --num_decoding_workers 1
```

### Supported Tasks

All tasks supported by lm-evaluation-harness are available, including:

- `hellaswag`: HellaSwag commonsense reasoning
- `mmlu`: Massive Multitask Language Understanding
- `truthfulqa`: TruthfulQA
- `arc`: ARC (AI2 Reasoning Challenge)
- `winogrande`: Winogrande
- And many more...

See [lm-evaluation-harness tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks) for full list.

## Implementation Details

### KVServeEvaluator

The `KVServeEvaluator` class implements the `LM` interface required by lm-evaluation-harness:

- `loglikelihood(requests)`: Compute log-likelihood of continuations
- `loglikelihood_rolling(requests)`: Compute full log-likelihood for perplexity
- `generate_until(requests)`: Generate text until stopping sequences

### Current Limitations

1. **Loglikelihood computation**: Currently returns placeholder values. Full implementation requires:
   - Access to model logprobs during generation
   - Proper handling of continuation token probabilities

2. **Loglikelihood rolling**: Placeholder implementation. Requires:
   - Chunking long sequences
   - Rolling window computation

3. **Batch processing**: Currently processes requests sequentially. Future optimization:
   - Batch multiple requests together
   - Parallel processing

## Future Improvements

- [ ] Implement proper loglikelihood computation using model logprobs
- [ ] Implement rolling window loglikelihood for perplexity
- [ ] Add batch processing support
- [ ] Add caching support
- [ ] Performance optimizations


