# Transformer

KV cache transformation module for compression in KVServe.

## Overview

The transformer module implements transformations for KV cache compression. It supports **Hadamard transform** to make the KV cache more compressible.

## Architecture

The transformation pipeline consists of two main components:

1. **Hadamard Transform Functions** (`hadamard_func.py`): Core Hadamard transformation operations
   - `HadamardTransform`: Fast Walsh-Hadamard Transform (FWHT) implementation
   - `transform`: Forward Hadamard transform with Rademacher signs
   - `inverse`: Inverse Hadamard transform to restore original data

2. **KVServe Transformer** (`kvserve_transformer.py`): Main transformer implementation
   - Implements `Transformer` interface from `kvserve.manager.components`
   - Coordinates transformation operations
   - Manages transformation configuration

## Usage

### Basic Usage with Compression Manager

```python
import torch
from kvserve.manager import CompressionManager, CompressionConfig
from kvserve.transformer import KVServeTransformer

# Generate random tensor for shape [k/v, num_blocks, block_size, num_heads, head_size]
kv_cache = torch.randn(2, 128, 32, 8, 128, dtype=torch.bfloat16, device="cuda")

# Configure compression with transformer
config = CompressionConfig(
    enabled=True,
    pipeline=["transformer"],
    transformer_config={
        "transform_type": "hadamard",
        "seed": 0x3333,
    },
    min_compress_size=1024,
)

# Create compression manager
manager = CompressionManager(
    config=config,
    transformer=KVServeTransformer,
)

# Transform KV cache
transformed = manager.transform(
    layer_id=0,
    kv_data=kv_cache,  # [2, num_blocks, block_size, num_heads, head_size]
    request_id="req_001",
    metadata={"block_indices": [0, 1, 2]},
)

# Inverse transform KV cache
restored = manager.inverse_transform(
    layer_id=0,
    transformed_data=transformed,
)

# Calculate cosine similarity between original and restored KV cache
cosine_similarity = torch.nn.functional.cosine_similarity(kv_cache.flatten(), restored.flatten(), dim=0)
print(f"Cosine similarity: {cosine_similarity.item()}")
```

## Configuration Parameters

### KVServeTransformer Parameters

- `transform_type` (str, default: `"hadamard"`): Type of transformation to apply.

- `seed` (int, default: `0x3333`): Base seed for generating Rademacher signs. The seed is combined with layer and head indices to generate unique random signs for each head in each layer.

## Transformation Process

### Forward Transform Pipeline

1. **Generate Rademacher Signs**: Generate random ±1 signs for each head using deterministic seed
   - Seed formula: `base_seed ^ (layer_idx << 16) ^ head_idx`
   - Signs are cached to avoid repeated CPU generation and Host-to-Device transfer

2. **Apply Signs**: Element-wise multiply KV cache with signs
   - Shape: `[num_blocks, block_size, num_heads, head_dim] * [1, 1, num_heads, head_dim]`

3. **Fast Walsh-Hadamard Transform (FWHT)**: Apply FWHT along the last dimension
   - If head_dim is not a power of two, split into power-of-two chunks
   - Scale factor: `1.0 / sqrt(chunk_size)`

### Inverse Transform Pipeline

1. **Apply FWHT**: Apply inverse Hadamard transform (same as forward due to symmetry)

2. **Apply Signs**: Multiply by Rademacher signs again to restore original values
   - Since Hadamard matrix is symmetric orthogonal and signs are their own inverse

## Components

### Hadamard Transform

- **`HadamardTransform.__init__(base_seed)`**: Initialize transformer with base seed for Rademacher sign generation

- **`transform(layer_id, tensor)`**: Apply forward Hadamard transform
  - Input: `[num_blocks, block_size, num_heads, head_dim]`
  - Output: Transformed tensor with same shape

- **`inverse(layer_id, tensor)`**: Apply inverse Hadamard transform
  - Input: Transformed tensor `[num_blocks, block_size, num_heads, head_dim]`
  - Output: Restored original tensor with same shape

- **`get_rademacher_signs(layer_idx, num_heads, head_dim, device, dtype)`**: Generate deterministic random signs
  - Returns: `[1, 1, num_heads, head_dim]` tensor for broadcasting
  - Signs are cached per layer/head configuration

- **`_fwht_in_chunks(x)`**: Apply FWHT to last dimension, handling non-power-of-two dimensions
  - Splits dimension into descending powers-of-two chunks
  - Applies transform to each chunk separately

## Dependencies

The transformer module requires:
- [fast_hadamard_transform](https://github.com/Dao-AILab/fast-hadamard-transform): Fast CUDA implementation of Hadamard transform
- PyTorch for tensor operations

