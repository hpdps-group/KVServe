import torch
from kvserve.manager import CompressionManager, CompressionConfig
from kvserve.quantizer import KVServeQuantizer

# Generate random tensor for shape [k/v, num_blocks, block_size, num_heads, head_size]
kv_cache = torch.rand(2, 128, 32, 8, 128, dtype=torch.bfloat16, device="cuda")

# Configure compression with quantizer
config = CompressionConfig(
    enabled=True,
    pipeline=["quantizer"],
    quantizer_config={
        "model_name": "Llama-3.1-8B-Instruct",
        "hybrid_ratio": 0.3,
        "high_key_max_value": 16,
        "high_value_max_value": 16,
        "low_key_max_value": 12,
        "low_value_max_value": 12,
        "axis_key": "channel",
        "axis_value": "token",
        "split_type": "head",
    },
    min_compress_size=1024,
)

# Create compression manager
manager = CompressionManager(
    config=config,
    quantizer=KVServeQuantizer,
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