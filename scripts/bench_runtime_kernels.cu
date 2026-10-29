// Benchmark entry points around the *existing* production CUDA kernels.
// Including the source keeps the kernel implementation identical to runtime.
#include "../kvserve_v1/compression/runtime/csrc/kvs_compress.cu"

void bench_fwht(int64_t input, int64_t output, int64_t scratch, int layers,
                int tokens, int heads, int dim, int64_t stream_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const int64_t n = static_cast<int64_t>(tokens) * heads * dim;
  for (int layer = 0; layer < layers; ++layer) {
    const bf16* src = reinterpret_cast<const bf16*>(input) + layer * n;
    bf16* a = reinterpret_cast<bf16*>(output) + layer * n;
    bf16* b = reinterpret_cast<bf16*>(scratch) + layer * n;
    bf16* dst = a;
    for (int order = 1; order < dim; order <<= 1) {
      fwht_stage_kernel<<<blocks_for(n), kThreads, 0, stream>>>(src, dst, dim,
                                                               order, n);
      check_launch("bench_fwht");
      src = dst;
      dst = dst == a ? b : a;
    }
    scale_kernel<<<blocks_for(n), kThreads, 0, stream>>>(
        const_cast<bf16*>(src), 1.0f / std::sqrt(static_cast<float>(dim)), n);
    check_launch("bench_scale");
    if (src != a) {
      cudaError_t status = cudaMemcpyAsync(a, src, n * sizeof(bf16),
                                          cudaMemcpyDeviceToDevice, stream);
      if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
  }
}

void bench_quant(int64_t input, int64_t output, int64_t minimum, int64_t scale,
                 int64_t maximum, int64_t inverse, int layers, int tokens,
                 int heads, int dim, int high_levels, int low_levels,
                 int low_layers, int axis, int64_t stream_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const int64_t n = static_cast<int64_t>(tokens) * heads * dim;
  const int aux = axis == 0 ? heads * dim : tokens;
  for (int layer = 0; layer < layers; ++layer) {
    const auto* src = reinterpret_cast<const bf16*>(input) + layer * n;
    auto* out = reinterpret_cast<uint8_t*>(output) + layer * n;
    auto* mn = reinterpret_cast<bf16*>(minimum) + layer * aux;
    auto* sc = reinterpret_cast<bf16*>(scale) + layer * aux;
    auto* mx = reinterpret_cast<bf16*>(maximum) + layer * aux;
    auto* inv = reinterpret_cast<bf16*>(inverse) + layer * aux;
    const int levels = layer >= layers - low_layers ? low_levels : high_levels;
    if (axis == 0) {
      quantize_channel_kernel<<<ceil_div(heads * dim, kChannelTile), kThreads, 0, stream>>>(
          src, out, mn, sc, mx, inv, tokens, heads, 0, heads, dim, levels);
    } else {
      quantize_token_kernel<<<ceil_div(tokens, kTokensPerBlock), kThreads, 0, stream>>>(
          src, out, mn, sc, mx, inv, tokens, heads, 0, heads, dim, levels);
    }
    check_launch("bench_quant");
  }
}

void bench_dequant(int64_t input, int64_t output, int64_t minimum,
                   int64_t scale, int layers, int tokens, int heads, int dim,
                   int axis, int64_t stream_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const int64_t n = static_cast<int64_t>(tokens) * heads * dim;
  const int aux = axis == 0 ? heads * dim : tokens;
  for (int layer = 0; layer < layers; ++layer) {
    const auto* in = reinterpret_cast<const uint8_t*>(input) + layer * n;
    auto* out = reinterpret_cast<bf16*>(output) + layer * n;
    const auto* mn = reinterpret_cast<const bf16*>(minimum) + layer * aux;
    const auto* sc = reinterpret_cast<const bf16*>(scale) + layer * aux;
    if (axis == 0) {
      dequantize_channel_kernel<<<blocks_for(heads * dim), kThreads, 0, stream>>>(
          in, out, mn, sc, tokens, heads, 0, heads, dim);
    } else {
      dequantize_token_kernel<<<blocks_for(tokens), kThreads, 0, stream>>>(
          in, out, mn, sc, tokens, heads, 0, heads, dim);
    }
    check_launch("bench_dequant");
  }
}
