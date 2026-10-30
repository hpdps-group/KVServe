"""Fused transform numerics, stream ordering and worker-thread contracts."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from kvserve_v1.compression.runtime import cpp_backend
from kvserve_v1.compression.runtime.pipeline import CompressionPipeline

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def bf16_reference(source):
    result = source.clone()
    dim = result.shape[-1]
    step = 1
    while step < dim:
        pair = result.view(-1, dim // (2 * step), 2, step)
        a, b = pair[:, :, 0, :], pair[:, :, 1, :]
        result = torch.stack((a + b, a - b), dim=2).reshape_as(result)
        step *= 2
    return result.mul_(dim**-0.5)


@pytest.mark.parametrize("dim", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("rows", [1, 7, 33, 1025])
def test_transform_exact_bf16_reference(dim, rows):
    torch.manual_seed(42)
    source = torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16) * 17
    expected = bf16_reference(source)
    output = torch.empty_like(source)
    cpp_backend.hadamard_transform(source, output)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_default_transform_in_workers_and_graph_without_humming(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "humming", None)
    monkeypatch.setitem(sys.modules, "humming.ops", None)
    cpp_backend.load_kvs_compress(torch.device("cuda", 0))
    source = torch.randn(1025, 128, device="cuda", dtype=torch.bfloat16)
    original = source.clone()
    expected = bf16_reference(source)
    torch.cuda.synchronize()

    def run(capture):
        with torch.cuda.device(0):
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                output = torch.empty_like(source)
                CompressionPipeline._hadamard_transform(source, output, 128)
                if capture:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        CompressionPipeline._hadamard_transform(source, output, 128)
                    graph.replay()
            stream.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, (False, False)))
        pool.submit(run, True).result()
    torch.testing.assert_close(source, original, rtol=0, atol=0)


def test_transform_rejects_overlapping_buffers():
    storage = torch.empty(129, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="overlap"):
        cpp_backend.hadamard_transform(storage[:128], storage[1:])


def test_transform_matches_independent_sylvester_matrix():
    matrix = torch.ones(1, 1, dtype=torch.float64)
    for _ in range(7):
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
        )
    # Small integers make every BF16 intermediate exact; this independently
    # checks butterfly ordering and normalization, without a paired inverse.
    source = torch.randint(-1, 2, (33, 128), device="cuda").to(torch.bfloat16)
    expected = ((source.cpu().double() @ matrix) / (128**0.5)).to(torch.bfloat16)
    output = torch.empty_like(source)
    cpp_backend.hadamard_transform(source, output)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
