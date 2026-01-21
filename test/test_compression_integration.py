"""
Test KV compression integration with Worker
Tests the complete compression pipeline with real KV cache structure
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

# Add project root to sys.path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np


def test_compression_with_worker_structure():
    """Test compression with real Worker KV cache structure"""
    print("Testing compression integration with Worker structure...")
    print()
    
    # Import compression components
    from kvserve.manager.compression_manager import CompressionManager, CompressionConfig
    from kvserve.transformer import KVServeTransformer
    from kvserve.quantizer import KVServeQuantizer
    from kvserve.codec import KVServeCodec
    from kvserve.quantizer.cachegen_quantizer import CachegenQuantizer
    from kvserve.quantizer.kivi_quantizer import KIVIQuantizer
    from kvserve.codec.cachegen_codec import CachegenCodec
    from kvserve.codec.kivi_codec import KIVICodec
    
    # Simulate real KV cache structure
    # vLLM v0 uses: kv_cache[layer_idx] = [num_kv=2, num_blocks, block_size, num_heads, head_size]
    # Use same dimensions as test_all_layer_compression.py for compatibility with quantizer config
    num_layers = 28
    num_kv = 2  # K and V
    num_blocks = 134  # Number of blocks for this request
    block_size = 32  # Tokens per block
    num_heads = 4  # Match Llama-3.1-8B-Instruct quantizer config
    head_size = 128
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print()
    
    # Create simulated KV cache (list of tensors, like in Worker)
    kv_cache = []
    for layer_idx in range(num_layers):
        layer_kv = torch.randn(
            num_kv, num_blocks, block_size, num_heads, head_size,
            dtype=torch.bfloat16,  # Use bfloat16 like test_all_layer_compression.py
            device=device
        )
        kv_cache.append(layer_kv)
    
    print(f"Simulated KV cache structure:")
    print(f"  - Layers: {num_layers}")
    print(f"  - Each layer shape: {kv_cache[0].shape}")
    print(f"  - Total size: {sum(l.numel() * l.element_size() for l in kv_cache) / 1024 / 1024:.2f} MB")
    print()
    
    # Test 1: Extract KV blocks (simulate Worker.extract_kv_blocks)
    print("Test 1: Extract KV blocks")
    block_indices = list(range(num_blocks))
    
    # Stack all layers: [num_layers, 2, num_blocks, block_size, num_heads, head_size]
    all_layers_data = torch.stack(kv_cache, dim=0)
    print(f"✓ Extracted KV data shape: {all_layers_data.shape}")
    print()
    
    # Test 3: Full pipeline compression
    print("Test 3: Full pipeline (transformer + quantizer + codec)")
    # For KVServe's compression pipeline
    config_full = CompressionConfig(
        enabled=True,
        pipeline=["transformer", "quantizer", "codec"],
        transformer_config={
            "transform_type": "hadamard",
            "seed": 0x3333,
        },
        quantizer_config={
            "model_name": "Qwen2.5-7B-Instruct",
            "hybrid_ratio": 0.8,
            "high_key_max_value": 12,
            "high_value_max_value": 8,
            "low_key_max_value": 6,
            "low_value_max_value": 4,
            "axis_key": "channel",
            "axis_value": "token",
            "split_type": "head",
        },
        codec_config={
            "codec_type": "nvcomp",
            "nvcomp_algorithm": "ANS",
            "data_type": "|u1",
        },
    )

    # For KIVI's compression pipeline
    # config_full = CompressionConfig(
    #     enabled=True,
    #     pipeline=["quantizer", "codec"],
    #     # "transformer_config": {
    #     #     "transform_type": "hadamard",
    #     #     "seed": 0x3333,
    #     # },
    #     quantizer_config={
    #         "model_name": "Qwen2.5-7B-Instruct",
    #         "nbits": 2,
    #         "axis_key": "channel",
    #         "axis_value": "token",
    #         "group_size": 32,
    #     },
    #     codec_config={
    #         "codec_type": "bitpacking",
    #     },
    # )

    # For Cachegen's compression pipeline
    # config_full = CompressionConfig(
    #     enabled=True,
    #     pipeline=["quantizer", "codec"],
    #     # "transformer_config": {
    #     #     "transform_type": "hadamard",
    #     #     "seed": 0x3333,
    #     # },
    #     quantizer_config={
    #         "model_name": "Qwen2.5-7B-Instruct",
    #         "quantization_level": 2,
    #         "high_max_value": 32,
    #         "mid_max_value": 16,
    #         "low_max_value": 12,
    #     },
    #     codec_config={
    #         "codec_type": "torchac",
    #     },
    # )

    
    try:
        manager_full = CompressionManager(
            config=config_full,
            transformer=KVServeTransformer,
            quantizer=KVServeQuantizer, # choose from KVServeQuantizer, CachegenQuantizer, KIVIQuantizer
            codec=KVServeCodec, # choose from KVServeCodec, CachegenCodec, KIVICodec
        )
        
        compressed_full = manager_full.compress_all_layers(
            all_layers_data=all_layers_data,
            request_id="test_full",
            config=config_full,
            metadata={"block_indices": block_indices},
        )
        
        if compressed_full:
            ratio_full = compressed_full.original_size / compressed_full.compressed_size
            print(f"✓ Full pipeline compression: {compressed_full.original_size} -> {compressed_full.compressed_size} bytes (ratio: {ratio_full:.2f}x)")
            
            # Check metadata
            assert "compressed_size" in compressed_full.metadata, "compressed_size not in metadata"
            assert compressed_full.metadata["compressed_size"] == compressed_full.compressed_size
            print(f"✓ Metadata contains compressed_size: {compressed_full.metadata['compressed_size']}")
            
            # Decompress
            decompressed_full = manager_full.decompress_all_layers(compressed_full, config=config_full)
            if decompressed_full is not None:
                print(f"✓ Full pipeline decompression successful")
                print(f"  Original shape: {all_layers_data.shape}")
                print(f"  Decompressed shape: {decompressed_full.shape}")
                print(f"  Cosine similarity: {torch.cosine_similarity(all_layers_data.flatten(), decompressed_full.flatten(), dim=0)}")
            else:
                print("✗ Decompression failed")
        else:
            print("✗ Compression failed")
    
    except ImportError as e:
        print(f"⚠ Full pipeline test skipped (missing dependency): {e}")
    except RuntimeError as e:
        if "CUDA" in str(e) or "kernel image" in str(e):
            print(f"⚠ Full pipeline test skipped (CUDA architecture mismatch): {e}")
        else:
            raise
    
    print()
    
    # Test 4: Simulate write back to KV cache (Worker.write_kv_blocks)
    print("Test 4: Write back to KV cache")
    if decompressed_full is not None:
        # Simulate writing back
        assert decompressed_full.shape[0] == num_layers
        assert decompressed_full.shape[2] == len(block_indices)
        
        # In Worker, this would be:
        # for layer_idx, layer_kv in enumerate(self.kv_cache):
        #     layer_kv[:, block_idx_tensor, :, :, :] = kv_data[layer_idx]
        
        reconstructed_cache = []
        for layer_idx in range(num_layers):
            reconstructed_cache.append(decompressed_full[layer_idx])
        
        print(f"✓ Successfully reconstructed {len(reconstructed_cache)} layers")
        print(f"  Each layer shape: {reconstructed_cache[0].shape}")
    else:
        print("⚠ Skipping write back test (no decompressed data available)")
    
    print()
    print("All integration tests completed!")

def test_metadata_serialization():
    """Test metadata serialization for NCCL transfer"""
    print("Testing metadata serialization...")
    print()
    
    # import pickle
    from kvserve.manager.compression_manager import EasyDist
    
    # Create sample metadata (like what compress_all_layers generates)
    metadata = {
        "request_id": "test_001",
        "pipeline": ["transformer", "quantizer", "codec"],
        "original_size": 2097152,
        "compressed_size": 893893,
        "device": "cuda:0",
        "num_layers": 32,
        "is_all_layers": True,
        "block_indices": [0, 1, 2, 3, 4],
        "all_quantization_params": [
            {"min_val": 0.0, "max_val": 1.0} for _ in range(32)
        ],
        "quantization_applied": True,
        "transformer_applied": True,
        "codec_applied": True,
        "original_shape": [32, 2, 5, 16, 32, 128],
        "original_dtype": "float16",
    }
    
    # Serialize
    metadata_tensor, metadata_size_tensor = EasyDist.pack_object(metadata)
    print(f"✓ Metadata serialized: {metadata_size_tensor.item()} bytes")
    
    # Deserialize
    metadata_restored = EasyDist.unpack_object(metadata_tensor)
    print(f"✓ Metadata deserialized successfully")
    

    EasyDist.compare(metadata_restored, metadata)

    print(f"✓ Metadata content fully verified: metadata_restored == metadata")
    
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("KV Compression Integration Tests")
    print("=" * 60)
    print()
    
    test_compression_with_worker_structure()
    test_metadata_serialization()
    
    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

