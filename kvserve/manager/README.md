# Compression Manager

KV cache compression framework for PD separation.

## Architecture

The compression pipeline consists of three optional components:

1. **Transformer**: Transforms KV cache data (e.g., DCT, PCA)
2. **Quantizer**: Quantizes data to reduce precision
3. **Codec**: Applies codec compression (e.g., zlib, lz4)

## Usage

```python
from kvserve.manager import CompressionManager, CompressionConfig

# Configure compression
config = CompressionConfig(
    enabled=True,
    pipeline=["quantizer"],  # or ["transformer", "quantizer", "codec"]
    quantizer_config={"bits": 8},
    min_compress_size=1024,
)

# Initialize components (implement Transformer/Quantizer/Codec)
# my_transform = MyTransform()
# my_quantizer = MyQuantizer()
# my_codec = MyCodec()

# Create manager
manager = CompressionManager(
    config=config,
    # transformer=my_transformer,
    quantizer=my_quantizer,
    # codec=my_codec,
)

# Compress
compressed = manager.compress(
    kv_data=tensor,  # torch.Tensor
    request_id="req_001",
    metadata={"block_indices": [0, 1, 2]},
)

# Decompress
decompressed = manager.decompress(compressed)
```

## Pipeline

The compression pipeline processes data in order:

1. **Compress**: Transformer(transform) → Quantizer(quantize) → Codec(encode) → Transfer
2. **Decompress**: Transfer → Codec(decode) → Quantizer(dequantize) → Transformer(reverse)

## Components

Implement the abstract base classes:
- `Transformer`: `transform()` and `reverse()` methods
- `Quantizer`: `quantize()` and `dequantize()` methods  
- `Codec`: `encode()` and `decode()` methods


