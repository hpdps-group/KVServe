# Codec

KV cache codec compression module for lossless/lossy compression in KVServe.

## Overview

The codec module implements lossless/lossy compression algorithms for KV cache compression. It supports **nvCOMP** compression library (other codecs are supported in the future) with multiple algorithms to achieve high compression ratios while preserving exact data integrity.

## Architecture

The codec compression pipeline consists of two main components:

1. **nvCOMP Functions** (`nvcomp_func.py`): Core nvCOMP compression operations
   - `nvCOMPCodec`: Wrapper for NVIDIA nvCOMP compression library
   - `encode`: Compress tensor to bytes using nvCOMP
   - `decode`: Decompress bytes back to tensor using nvCOMP

2. **KVServe Codec** (`kvserve_codec.py`): Main codec implementation
   - Implements `Codec` interface from `kvserve.manager.components`
   - Coordinates compression operations
   - Manages codec configuration

## Usage

### Basic Usage with Compression Manager

```python
import torch
from kvserve.manager import CompressionManager, CompressionConfig
from kvserve.codec import KVServeCodec

# Generate random tensor for shape [k/v, num_blocks, block_size, num_heads, head_size]
kv_cache = torch.randn(2, 128, 32, 8, 128, dtype=torch.bfloat16, device="cuda")

# Configure compression with codec
config = CompressionConfig(
    enabled=True,
    pipeline=["codec"],
    codec_config={
        "codec_type": "nvcomp",
        "nvcomp_algorithm": "ANS",
        "data_type": "|u1",
    },
    min_compress_size=1024,
)

# Create compression manager
manager = CompressionManager(
    config=config,
    codec=KVServeCodec,
)

# Compress KV cache
compressed = manager.compress(
    layer_id=0,
    kv_data=kv_cache,  # [2, num_blocks, block_size, num_heads, head_size]
    request_id="req_001",
    metadata={"block_indices": [0, 1, 2]},
)

# Decompress KV cache
decompressed = manager.decompress(
    layer_id=0,
    compressed_data=compressed,
)

# Calculate cosine similarity between original and decompressed KV cache
cosine_similarity = torch.nn.functional.cosine_similarity(kv_cache.flatten(), decompressed.flatten(), dim=0)
print(f"Cosine similarity: {cosine_similarity.item()}")
```

## Configuration Parameters

### KVServeCodec Parameters

- `codec_type` (str, default: `"nvcomp"`): Type of codec to use. Currently only supports `"nvcomp"`.

- `nvcomp_algorithm` (str, default: `"ANS"`): Compression algorithm to use. Supported algorithms:
  - `"ANS"`: Asymmetric Numeral Systems - Fast entropy coding
  - `"Bitcomp"`: Bit compression algorithm
  - `"Cascaded"`: Cascaded compression
  - `"Deflate"`: Deflate compression (zlib-compatible)
  - `"GDeflate"`: GPU-accelerated Deflate compression
  - `"LZ4"`: LZ4 fast compression
  - `"Zstd"`: Zstandard compression

## Compression Process

### Encoding Pipeline

1. **Flatten Tensor**: Convert tensor to flattened uint8 view
   - Shape: `[num_blocks, block_size, num_heads, head_size]` -> flattened uint8 array

2. **Convert to nvCOMP Array**: Convert PyTorch tensor to nvCOMP array format

3. **Compress**: Apply nvCOMP compression algorithm
   - Output: Compressed bytes buffer

4. **Convert to Bytes**: Convert compressed buffer to Python bytes object

### Decoding Pipeline

1. **Convert from Bytes**: Convert compressed bytes to CuPy array

2. **Convert to nvCOMP Array**: Convert CuPy array to nvCOMP array format

3. **Decompress**: Apply nvCOMP decompression algorithm
   - Output: Decompressed buffer

4. **Restore Tensor**: Convert decompressed buffer back to PyTorch tensor
   - Restore original dtype and shape
   - Place tensor on specified device

## Components

### nvCOMP Codec

- **`nvCOMPCodec.__init__(algorithm)`**: Initialize nvCOMP codec with specified algorithm

- **`encode(tensor)`**: Compress tensor to bytes
  - Input: PyTorch tensor of any shape and dtype
  - Output: Compressed bytes
  - Process: Flatten -> uint8 view -> nvCOMP compress -> bytes

- **`decode(compressed_data, original_dtype, original_shape, device)`**: Decompress bytes to tensor
  - Input: Compressed bytes, original dtype string, original shape list, target device
  - Output: Decompressed PyTorch tensor with original dtype and shape
  - Process: bytes -> CuPy array -> nvCOMP decompress -> PyTorch tensor -> restore dtype/shape

## Dependencies

The codec module requires:
- [nvCOMP](https://docs.nvidia.com/cuda/nvcomp/py_api.html#): NVIDIA GPU-accelerated compression library
- PyTorch for tensor operations

