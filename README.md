# KV Serve - PD Separation Engine

A simplified Prefill-Decode separation engine based on vLLM, extracted from ElasticMM project.

## Overview

This project implements Prefill-Decode (PD) separation for efficient LLM serving:
- **Prefill Stage**: Processes full input sequences and generates KV cache
- **Decode Stage**: Handles autoregressive token generation
- **KV Transfer**: Transfers KV cache from Prefill to Decode via NCCL P2P

## Architecture

```
Request → Prefill Engine → KV Transfer → Decode Engine → Output
                ↓                             ↓
          Generate KV Cache          Autoregressive Generation
```

## Key Components

1. **`engine/utils.py`**: Core data structures (Request, StepOutput, etc.)
2. **`engine/block_manager.py`**: KV cache block management
3. **`engine/worker.py`**: Ray worker for vLLM model execution
4. **`engine/worker_steps.py`**: Prefill and decode step implementations
5. **`engine/stage_engine.py`**: PrefillEngine and DecodeEngine
6. **`engine/kv_transfer.py`**: KV cache transfer between stages
7. **`engine/backend.py`**: Backend coordinating Prefill and Decode stages

## Requirements

- Python 3.8+
- Ray 2.49.1
- vLLM 0.10.1+
- PyTorch 2.8.0+ (with CUDA support)
- transformers 4.56.1+
- Flash Attention 2.8.1+

## Installation

1. Install PyTorch with CUDA support:
```bash
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

2. Install vLLM (if using from source):
```bash
pip install -e /path/to/vllm-0.10.1
```

3. Install other dependencies:
```bash
pip install -r requirements.txt
```

Note: Some packages like `flash_attn` may require compilation. See individual package documentation for installation details.

## Usage

### Basic Example

```python
import asyncio
import ray
from kvserve.engine.backend import PDBackend
from kvserve.engine.utils import Request

async def main():
    # Initialize Ray
    ray.init(num_gpus=2)
    
    # Create backend
    backend = PDBackend(
        model_path="/path/to/model",
        num_prefill_workers=1,
        num_decoding_workers=1,
        block_size=16,
        max_num_gpu_blocks=3000,
        kv_transfer_method="nccl",
    )
    
    # Initialize and start
    await backend.initialize()
    await backend.start()
    
    # Create and submit request
    request = Request(
        request_id="test_001",
        prompt="Hello, how are you?",
        prompt_token_ids=[101, 7592, 1010, 2129, 2024, 2017, 102],
        max_tokens=50,
    )
    
    await backend.add_request(request)
    
    # Collect outputs
    while True:
        outputs = await backend.get_outputs()
        for output in outputs:
            if output.finished:
                print(f"Completed: {output.request_id}")
                break
        await asyncio.sleep(0.1)
    
    # Stop backend
    await backend.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Running Tests

```bash
# Update model_path in test/test_pd_separation.py
python test/test_pd_separation.py
```

## Notes

- This is a simplified version extracted from ElasticMM's V0 backend
- Only supports Prefill-Decode separation (no Encoding stage)
- Requires at least 2 GPUs (1 for Prefill, 1 for Decode)
- KV transfer uses NCCL P2P by default


