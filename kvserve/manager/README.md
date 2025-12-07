# Compression Manager

KV cache compression framework for PD separation.

## Architecture

The compression pipeline consists of three optional components:

1. **Transform**: Transforms KV cache data (e.g., DCT, PCA)
2. **Quantizer**: Quantizes data to reduce precision
3. **Lossless Compression**: Applies lossless compression (e.g., zlib, lz4)

## Usage

```python
from kvserve.manager import CompressionManager, CompressionConfig

# Configure compression
config = CompressionConfig(
    enabled=True,
    pipeline=["quantizer"],  # or ["transform", "quantizer", "lossless"]
    quantizer_config={"bits": 8},
    min_compress_size=1024,
)

# Initialize components (implement Transform/Quantizer/LosslessCompression)
# my_transform = MyTransform()
# my_quantizer = MyQuantizer()
# my_lossless = MyLosslessCompression()

# Create manager
manager = CompressionManager(
    config=config,
    # transform=my_transform,
    quantizer=my_quantizer,
    # lossless=my_lossless,
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

1. **Compress**: Transform → Quantize → Lossless → Transfer
2. **Decompress**: Transfer → Lossless → Dequantize → Transform (reverse)

## Components

Implement the abstract base classes:
- `Transform`: `encode()` and `decode()` methods
- `Quantizer`: `quantize()` and `dequantize()` methods  
- `LosslessCompression`: `compress()` and `decompress()` methods

