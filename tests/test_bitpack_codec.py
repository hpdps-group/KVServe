"""Exact GPU codec contracts, modular pipeline and CUDA stream checks."""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

pytest.importorskip("triton")
from kvserve_v1.compression.codec import BitPackCodec, KVServeCodec


@pytest.mark.parametrize("bits", [0, 9, True, 3.5, None])
def test_invalid_width(bits):
    with pytest.raises(ValueError, match="bitpack_bits"):
        BitPackCodec(bitpack_bits=bits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("shape", [[], [0, 2], [1], [7], [13], [2, 3, 257], [8193]])
def test_exact_roundtrip(bits, shape):
    codec = KVServeCodec(codec_type="bitpack", bitpack_bits=bits)
    x = torch.randint(2**bits, shape, device="cuda", dtype=torch.uint8)
    packed = codec.encode(0, x)
    assert packed.numel() == 16 + (x.numel() * bits + 7) // 8
    assert torch.equal(x, codec.decode(0, packed, "uint8", shape, str(x.device)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_noncontiguous_stream_graph_and_packet_isolation():
    codec = BitPackCodec(bitpack_bits=3)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randint(8, (17, 13), device="cuda", dtype=torch.uint8).t()
        a = codec.encode(x)
        b = codec.encode(x + 0)
        restored = codec.decode(a, "uint8", list(x.shape), str(x.device))
        assert torch.equal(x, restored)
        assert a.data_ptr() != b.data_ptr()
        # Warm all kernels before capture, including the contiguous-input path.
        source = x.contiguous()
        codec.decode(codec.encode(source), "uint8", list(x.shape), str(x.device))
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        packed = codec.encode(source)
        restored = codec.decode(packed, "uint8", list(x.shape), str(x.device))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(source, restored)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_concurrent_per_call_widths_do_not_change_default():
    codec = BitPackCodec()
    def roundtrip(bits):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.randint(2**bits, (127,), device="cuda", dtype=torch.uint8)
            packet = codec.encode(x, bitpack_bits=bits)
            restored = codec.decode(packet, "uint8", [127], "cuda", bitpack_bits=bits)
        stream.synchronize()
        return x, restored
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert all(torch.equal(x, y) for x, y in pool.map(roundtrip, [2, 4, 7]))
    assert codec.bitpack_bits == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_invalid_host_contracts():
    codec = BitPackCodec(bitpack_bits=3)
    with pytest.raises(ValueError, match="CUDA"):
        codec.encode(torch.zeros(1, dtype=torch.uint8))
    with pytest.raises(TypeError, match="uint8"):
        codec.encode(torch.zeros(1, device="cuda"))
    x = torch.zeros(13, device="cuda", dtype=torch.uint8)
    packed = codec.encode(x)
    with pytest.raises(ValueError, match="length"):
        codec.decode(packed[:-1], "uint8", [13], "cuda")
    with pytest.raises(TypeError, match="uint8"):
        codec.decode(packed, "float16", [13], "cuda")
    with pytest.raises(ValueError, match="nonnegative"):
        codec.decode(packed, "uint8", [-1], "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_modular_manager_wire_roundtrip():
    from kvserve_v1.compression.compression_manager import CompressionConfig, CompressionManager
    from kvserve_v1.compression.wire import build_wire, restore_from_wire
    manager = CompressionManager(
        CompressionConfig(enabled=True, pipeline=["codec"],
                          codec_config={"codec_type":"bitpack", "bitpack_bits":4}),
        codec=KVServeCodec)
    x = torch.randint(16, (2,2,3,16,2,128), device="cuda", dtype=torch.uint8)
    compressed = manager.compress_all_layers(x, "bitpack-wire", manager.config, {})
    assert compressed is not None and compressed.metadata["codec_applied"]
    restored = manager.decompress_all_layers(
        restore_from_wire(build_wire(compressed, max_chunk_bytes=4096)), manager.config)
    assert torch.equal(x, restored)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quantizer_pipeline_matches_unpacked_reference():
    from kvserve_v1.compression.manager import KVCompressionAdapter
    from kvserve_v1.compression.wire import build_wire, restore_from_wire
    config = {"pipeline":["quantizer", "codec"],
              "quantizer_config":{"model_name":"Qwen2.5-7B-Instruct",
                                   "high_key_max_value":12, "high_value_max_value":8,
                                   "low_key_max_value":6, "low_value_max_value":4},
              "codec_config":{"codec_type":"bitpack", "bitpack_bits":4}}
    packed = KVCompressionAdapter(config, num_kv_heads=4, head_size=128)
    reference = KVCompressionAdapter(
        {**config, "pipeline":["quantizer"]}, num_kv_heads=4, head_size=128)
    x = torch.randn(2,2,3,16,4,128, device="cuda", dtype=torch.bfloat16)
    packet = packed.compress(x, "quant-bitpack")
    assert packet is not None and packet.metadata["codec_applied"]
    restored = packed.decompress(restore_from_wire(build_wire(packet, max_chunk_bytes=4096)))
    expected = reference.decompress(reference.compress(x, "quant-only"))
    assert torch.equal(restored, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("invalid", ["range", "header"])
def test_gpu_contract_failure_is_not_silent(invalid):
    # Device assertions invalidate their CUDA context: test them in a child.
    code = """
import torch
from kvserve_v1.compression.codec import BitPackCodec
c = BitPackCodec(bitpack_bits=3)
x = torch.tensor([255 if INVALID == 'range' else 1], device='cuda', dtype=torch.uint8)
p = c.encode(x)
if INVALID == 'header':
    p[0] = 0
    c.decode(p, 'uint8', [1], 'cuda')
torch.cuda.synchronize()
""".replace("INVALID", repr(invalid))
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "device-side assert" in result.stderr
    expected = "symbol exceeds" if invalid == "range" else "invalid bitpack header"
    assert expected in result.stderr
