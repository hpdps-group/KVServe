// Benchmark entry points around the *existing* production CUDA kernels.
// Including the source keeps the kernel implementation identical to runtime.
#include "../kvserve_v1/compression/runtime/csrc/kvs_compress.cu"

void bench_fwht(int64_t input, int64_t output, int64_t scratch, int layers,
                int tokens, int heads, int dim, int64_t stream_ptr) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  const int64_t n = static_cast<int64_t>(tokens) * heads * dim;
  for (int layer = 0; layer < layers; ++layer) {
    launch_fwht(reinterpret_cast<const bf16*>(input) + layer * n,
                reinterpret_cast<bf16*>(output) + layer * n,
                reinterpret_cast<bf16*>(scratch) + layer * n,
                dim, static_cast<int64_t>(tokens) * heads, stream);
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
