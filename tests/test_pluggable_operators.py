"""GPU numerics, packet independence, and real modular pipeline roundtrips."""

from concurrent.futures import ThreadPoolExecutor

import msgpack
import pytest
import torch

pytest.importorskip("triton")
from kvserve_v1.compression.codec import KVServeCodec
from kvserve_v1.compression.compression_manager import (
    CompressionConfig,
    CompressionManager,
)
from kvserve_v1.compression.manager import KVCompressionAdapter
from kvserve_v1.compression.quantizer import CenteredCodebookQuantizer
from kvserve_v1.compression.quantizer.centered_codebook import _centroids
from kvserve_v1.compression.transformer import KVServeTransformer, OrthogonalTransform
from kvserve_v1.compression.transformer.orthogonal import _base, _signs
from kvserve_v1.compression.wire import build_wire, restore_from_wire

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def relative_mse(actual, expected):
    return (
        (actual.float() - expected.float()).square().sum()
        / expected.float().square().sum().clamp_min(1.0e-20)
    ).item()


@pytest.mark.parametrize("kind", ["dct", "haar"])
@pytest.mark.parametrize(
    "dim,dtype",
    [
        (17, torch.float32),
        (128, torch.bfloat16),
        (256, torch.float16),
        (512, torch.bfloat16),
    ],
)
def test_rotation_matches_orthogonal_reference_and_preserves_layout(kind, dim, dtype):
    op = OrthogonalTransform(kind, seed=321)
    x = torch.randn((2, 3, 17, 4, dim), device="cuda", dtype=dtype)
    matrix = (
        _signs(kind, dim, 4, 321, 7).float()[:, :, None]
        * _base(kind, dim, 321, 7)[None]
    ).cuda()
    torch.testing.assert_close(
        matrix.double().transpose(-1, -2) @ matrix.double(),
        torch.eye(dim, device="cuda", dtype=torch.float64).expand_as(matrix),
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    expected = torch.einsum("pbthd,hde->pbthe", x.double(), matrix.double()).to(dtype)
    actual = op.transform(7, x)
    assert actual.shape == x.shape and actual.dtype == dtype
    assert relative_mse(actual, expected) < 3.0e-7
    assert relative_mse(op.inverse(7, actual), x) < (
        2.0e-5 if dtype == torch.bfloat16 else 1.0e-6
    )
    # Token/head strided views must use the same logical layout.
    view = x[:, :, ::2]
    assert torch.equal(op.transform(7, view), op.transform(7, view.contiguous()))
    with torch.device("cuda"):
        assert _base(kind, dim, 321, 999).device.type == "cpu"
        assert _signs(kind, dim, 4, 321, 999).device.type == "cpu"


@pytest.mark.parametrize(
    "bits,refinements", [(1, 1), (2, 1), (3, 0), (3, 1), (3, 2), (4, 1)]
)
@pytest.mark.parametrize("mean_dtype", ["float32", "bfloat16"])
def test_codebook_bins_and_reconstruction_match_reference(
    bits, refinements, mean_dtype
):
    op = CenteredCodebookQuantizer(
        key_bits=bits,
        value_bits=bits,
        mean_dtype=mean_dtype,
        scale_refinements=refinements,
    )
    x = torch.randn((2, 1, 257, 3, 128), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x, dtype=torch.uint8)
    q, meta = op.quantize(0, x, out=out)
    assert q.data_ptr() == out.data_ptr()
    centers = torch.tensor(_centroids(bits), device="cuda")
    bounds = (centers[:-1] + centers[1:]) / 2
    restored = op.dequantize(0, q, meta)
    for part in (0, 1):
        values = x[part].reshape(-1, 3, 128).permute(1, 0, 2).float()
        mean, scale = [meta["parts"][part][k].float() for k in ("mean", "scale")]
        if part == 0:
            groups = torch.arange(257, device="cuda") // 128
            mean, scale = mean[:, groups], scale[:, groups]
        else:
            mean, scale = mean[:, :, None], scale[:, :, None]
        expected = torch.bucketize(
            ((values - mean) / scale.clamp_min(1.0e-20)).contiguous(), bounds
        ).byte()
        actual = q[part].reshape(-1, 3, 128).permute(1, 0, 2)
        assert torch.equal(actual, expected)
        expected = (
            (centers[expected.long()] * scale + mean)
            .permute(1, 0, 2)
            .reshape_as(x[part])
            .to(x.dtype)
        )
        assert torch.equal(restored[part], expected)


def test_packet_statistics_survive_other_requests_and_config_updates():
    encoder = CenteredCodebookQuantizer()
    decoder = CenteredCodebookQuantizer()
    first = torch.randn((2, 1, 129, 2, 128), device="cuda", dtype=torch.bfloat16) + 100
    q, meta = encoder.quantize(0, first)
    expected = decoder.dequantize(0, q, meta)
    encoder.quantize(0, first * 100, key_bits=2, value_bits=4, group_size=64)
    decoder.update_params(key_bits=1, group_size=256, mean_dtype="bfloat16")
    assert torch.equal(decoder.dequantize(0, q, meta), expected)
    broken = {
        **meta,
        "parts": [
            {**meta["parts"][0], "mean": torch.empty(1, device="cuda")},
            meta["parts"][1],
        ],
    }
    with pytest.raises(ValueError, match="auxiliary"):
        decoder.dequantize(0, q, broken)


@pytest.mark.parametrize("kind", ["dct", "haar"])
def test_streams_and_cuda_graph_have_independent_outputs(kind):
    transform = OrthogonalTransform(kind)
    quantizer = CenteredCodebookQuantizer()

    def operation(x):
        y = transform.transform(0, x)
        q, meta = quantizer.quantize(0, y)
        return transform.inverse(0, quantizer.dequantize(0, q, meta))

    x = torch.randn((2, 1, 257, 4, 128), device="cuda", dtype=torch.bfloat16)
    operation(x)
    ready = torch.cuda.Event()
    ready.record()

    def run(_):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            result = operation(x)
        stream.synchronize()
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(run, range(2)))
    assert a.data_ptr() != b.data_ptr() and torch.equal(a, b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = operation(x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, a)


@pytest.mark.parametrize("kind", ["dct", "haar"])
@pytest.mark.parametrize("chunked", [False, True])
def test_separate_managers_restore_wire_and_chunk_layer_ids(kind, chunked):
    cfg = CompressionConfig(
        enabled=True,
        pipeline=["transformer", "quantizer", "codec"],
        transformer_config={"transform_type": kind, "seed": 123},
        quantizer_config={"impl": "centered_codebook"},
        codec_config={"codec_type": "bitpack", "bitpack_bits": 3},
    )

    def manager():
        return CompressionManager(
            cfg, KVServeTransformer, CenteredCodebookQuantizer, KVServeCodec
        )

    encoder, decoder = manager(), manager()
    shape = (9, 2, 256, 128, 4, 128) if chunked else (3, 2, 2, 17, 4, 128)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    packet = encoder.compress_all_layers(x, "operators-wire", cfg, {})
    assert packet is not None and packet.is_chunked == chunked
    wire = build_wire(packet, max_chunk_bytes=1 << 20)
    wire.meta = msgpack.unpackb(msgpack.packb(wire.meta), raw=False)
    restored = decoder.decompress_all_layers(restore_from_wire(wire), cfg)
    if chunked:
        restored = torch.cat(restored)
    assert restored.shape == x.shape and restored.dtype == x.dtype
    assert relative_mse(restored, x) < 0.06


def test_empty_and_constant_groups_are_finite():
    op = CenteredCodebookQuantizer()
    for shape in [(2, 0, 16, 4, 128), (2, 1, 17, 4, 128)]:
        x = torch.full(shape, 100.0, device="cuda", dtype=torch.bfloat16)
        q, meta = op.quantize(0, x)
        assert torch.equal(op.dequantize(0, q, meta), x)
    backing = torch.empty((2, 1, 17, 4, 256), device="cuda", dtype=torch.bfloat16)
    view = backing[..., ::2]
    alias = backing.view(torch.uint8).flatten()[: view.numel()].view(view.shape)
    with pytest.raises(ValueError, match="independent"):
        op.quantize(0, view, out=alias)


@pytest.mark.parametrize("kind", ["dct", "haar"])
def test_public_adapter_selects_new_components(kind):
    cfg = dict(
        pipeline=["transformer", "quantizer", "codec"],
        transformer_config=dict(transform_type=kind),
        quantizer_config=dict(impl="centered_codebook"),
        codec_config=dict(codec_type="bitpack", bitpack_bits=3),
    )
    encoder = KVCompressionAdapter(cfg, num_kv_heads=4, head_size=128)
    decoder = KVCompressionAdapter(cfg, num_kv_heads=4, head_size=128)
    x = torch.randn((3, 2, 2, 17, 4, 128), device="cuda", dtype=torch.bfloat16)
    packet = encoder.compress(x, "public-operators")
    assert packet is not None
    restored = decoder.decompress(
        restore_from_wire(build_wire(packet, max_chunk_bytes=1 << 20))
    )
    assert restored.shape == x.shape and relative_mse(restored, x) < 0.06
    with pytest.raises(ValueError, match="cannot use"):
        KVCompressionAdapter({**cfg, "impl": "tilelang_fused"}, 4, 128)
