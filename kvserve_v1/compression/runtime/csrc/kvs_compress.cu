// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// CUDA implementation of the Mooncake compression encode/decode path.
//
// All per-layer work (gather, FWHT, quantize, dequantize, scatter, tail) is
// driven from C++ with one Python boundary crossing per chunk.  Numerics
// mirror compression/pipeline.py: every bf16 operation rounds once, the
// transform runs the same seven butterfly stages and quantize follows the
// same op order, so outputs are byte-identical.

#include "kvs_compress.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <stdexcept>

namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kChannelTile = 8;
constexpr int kTokensPerBlock = kThreads / kWarpSize;

__device__ __forceinline__ float to_f32(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 rn(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ bf16 load_bf16(const char* p) {
  return *reinterpret_cast<const bf16*>(p);
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
int blocks_for(int64_t total) {
  return static_cast<int>(ceil_div(total, kThreads));
}

int tail_count_host(int tail_tokens, int kernel_tokens, int block) {
  const int remaining = tail_tokens - block * kernel_tokens;
  if (remaining <= 0) return 0;
  return remaining < kernel_tokens ? remaining : kernel_tokens;
}

void check_launch(const char* what) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("kvs_compress: ") + what +
                             " launch failed: " + cudaGetErrorString(err));
  }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// out[p, t, h, d] = cache[block_indices[p], head_order ? head_order[h] : h, t,
// d]
//                  * signs[h, d]
__global__ void gather_sign_kernel(
    const char* __restrict__ cache_base, int64_t cache_block_stride,
    int64_t cache_head_stride, int64_t cache_token_stride,
    const int64_t* __restrict__ block_indices,
    const int64_t* __restrict__ head_order, const void* __restrict__ signs,
    int num_heads, int head_size, int kernel_tokens, int64_t total,
    bf16* __restrict__ out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int d = static_cast<int>(idx % head_size);
  int64_t rest = idx / head_size;
  const int h = static_cast<int>(rest % num_heads);
  rest /= num_heads;
  const int t = static_cast<int>(rest % kernel_tokens);
  const int64_t p = rest / kernel_tokens;
  const int64_t head = head_order ? head_order[h] : h;
  const char* src = cache_base + block_indices[p] * cache_block_stride +
                    head * cache_head_stride + t * cache_token_stride +
                    static_cast<int64_t>(d) * 2;
  float value = to_f32(load_bf16(src));
  if (signs) {
    value *= to_f32(load_bf16(reinterpret_cast<const char*>(signs) +
                              (static_cast<int64_t>(h) * head_size + d) * 2));
  }
  out[idx] = rn(value);
}

// One FWHT butterfly stage (torch.add/torch.sub semantics on bf16).
__global__ void fwht_stage_kernel(const bf16* __restrict__ src,
                                  bf16* __restrict__ dst, int head_size,
                                  int order, int64_t total) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t row = idx / head_size;
  const int j = static_cast<int>(idx % head_size);
  const int group = j & ~(2 * order - 1);
  const int offset = j & (order - 1);
  const float a = to_f32(src[row * head_size + group + offset]);
  const float b = to_f32(src[row * head_size + group + order + offset]);
  const float value = (j & order) == 0 ? a + b : a - b;
  dst[row * head_size + j] = rn(value);
}

__global__ void scale_kernel(bf16* __restrict__ data, float scale,
                             int64_t total) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  data[idx] = rn(to_f32(data[idx]) * scale);
}

// Keep each row in registers through all butterfly stages. A subgroup of
// min(D, 32) lanes owns a row; larger rows use multiple registers per lane.
// Rounding after every stage preserves the existing BF16 wire numerics.
template <int D>
__global__ void fwht_fused_kernel(const bf16* __restrict__ src,
                                 bf16* __restrict__ dst, int64_t rows,
                                 float scale) {
  constexpr int width = D < kWarpSize ? D : kWarpSize;
  constexpr int values = D / width;
  const int lane = threadIdx.x % width;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * (kThreads / width) +
                      threadIdx.x / width;
  float v[values];
#pragma unroll
  for (int i = 0; i < values; ++i) {
    v[i] = row < rows ? to_f32(src[row * D + i * width + lane]) : 0.0f;
  }
  // All lanes, including inactive tail rows, participate in the shuffles.
#pragma unroll
  for (int step = 1; step < width; step <<= 1) {
#pragma unroll
    for (int i = 0; i < values; ++i) {
      const float other = __shfl_xor_sync(0xffffffff, v[i], step, width);
      v[i] = to_f32(rn((lane & step) ? other - v[i] : v[i] + other));
    }
  }
#pragma unroll
  for (int step = 1; step < values; step <<= 1) {
#pragma unroll
    for (int base = 0; base < values; base += 2 * step) {
#pragma unroll
      for (int j = 0; j < step; ++j) {
        const float a = v[base + j];
        const float b = v[base + step + j];
        v[base + j] = to_f32(rn(a + b));
        v[base + step + j] = to_f32(rn(a - b));
      }
    }
  }
  if (row < rows) {
#pragma unroll
    for (int i = 0; i < values; ++i) {
      dst[row * D + i * width + lane] = rn(v[i] * scale);
    }
  }
}

void launch_fwht(const bf16* input, bf16* output, bf16* scratch, int head_size,
                 int64_t rows, cudaStream_t stream) {
  if (head_size < 1 || (head_size & (head_size - 1))) {
    throw std::invalid_argument("FWHT requires a power-of-two head size");
  }
  if (rows <= 0) return;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
#define KVS_FWHT_CASE(D)                                                      \
  case D:                                                                    \
    fwht_fused_kernel<D><<<ceil_div(rows, kThreads / ((D) < 32 ? (D) : 32)),    \
                            kThreads, 0, stream>>>(input, output, rows, scale); \
    check_launch("fwht_fused");                                              \
    return
  switch (head_size) {
    KVS_FWHT_CASE(1);
    KVS_FWHT_CASE(2);
    KVS_FWHT_CASE(4);
    KVS_FWHT_CASE(8);
    KVS_FWHT_CASE(16);
    KVS_FWHT_CASE(32);
    KVS_FWHT_CASE(64);
    KVS_FWHT_CASE(128);
    KVS_FWHT_CASE(256);
    KVS_FWHT_CASE(512);
    KVS_FWHT_CASE(1024);
  }
#undef KVS_FWHT_CASE
  // Unusually large head sizes retain the previous two-buffer algorithm.
  const int64_t count = rows * head_size;
  const bf16* src = input;
  bf16* dst = output;
  for (int order = 1; order < head_size; order <<= 1) {
    fwht_stage_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
        src, dst, head_size, order, count);
    check_launch("fwht_stage");
    src = dst;
    dst = dst == output ? scratch : output;
  }
  scale_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      const_cast<bf16*>(src), scale, count);
  check_launch("fwht_scale");
  if (src != output) {
    const auto status = cudaMemcpyAsync(output, src, count * sizeof(bf16),
                                        cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
}

struct QuantParams {
  bf16 minimum;
  bf16 maximum;
  bf16 scale;
  bf16 inverse;
};

__device__ __forceinline__ QuantParams quant_params(float mn, float mx,
                                                   int levels) {
  const bf16 minimum = rn(mn);
  const bf16 maximum = rn(mx);
  float raw = to_f32(rn(to_f32(maximum) - to_f32(minimum)));
  if (raw < 1e-5f) raw = to_f32(rn(1e-5f));
  const bf16 scale = rn(raw / static_cast<float>(levels - 1));
  return {minimum, maximum, scale, rn(1.0f / to_f32(scale))};
}

__device__ __forceinline__ uint8_t quant_value(bf16 value, bf16 minimum,
                                             bf16 inverse, int levels) {
  float v = to_f32(value) - to_f32(minimum);
  v = to_f32(rn(v)) * to_f32(inverse);
  float q = rintf(to_f32(rn(v)));
  if (q < 0.0f) q = 0.0f;
  if (q > static_cast<float>(levels - 1)) q = static_cast<float>(levels - 1);
  return static_cast<uint8_t>(q);
}

// Eight adjacent channels share a block; 32 token lanes reduce each channel.
// Each warp reads four contiguous groups of eight values. Reduction and
// quantization share this tile, avoiding another launch or global scratch.
__global__ void quantize_channel_kernel(
    const bf16* __restrict__ src, uint8_t* __restrict__ out,
    bf16* __restrict__ aux_min, bf16* __restrict__ aux_scale,
    bf16* __restrict__ tmp_max, bf16* __restrict__ tmp_inv, int tokens,
    int heads, int head_start, int head_count, int head_size, int levels) {
  __shared__ float minima[kThreads / kWarpSize][kChannelTile];
  __shared__ float maxima[kThreads / kWarpSize][kChannelTile];
  const int c = threadIdx.x % kChannelTile;
  const int token_lane = threadIdx.x / kChannelTile;
  const int column = blockIdx.x * kChannelTile + c;
  const bool valid = column < head_count * head_size;
  const int h = head_start + column / head_size;
  const int d = column % head_size;
  float mn = INFINITY;
  float mx = -INFINITY;
  if (valid) {
    for (int t = token_lane; t < tokens; t += kThreads / kChannelTile) {
      const float v =
          to_f32(src[(static_cast<int64_t>(t) * heads + h) * head_size + d]);
      mn = fminf(mn, v);
      mx = fmaxf(mx, v);
    }
  }
  for (int offset = kChannelTile; offset < kWarpSize; offset <<= 1) {
    mn = fminf(mn, __shfl_xor_sync(0xffffffff, mn, offset));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, offset));
  }
  if (threadIdx.x % kWarpSize < kChannelTile) {
    minima[threadIdx.x / kWarpSize][c] = mn;
    maxima[threadIdx.x / kWarpSize][c] = mx;
  }
  __syncthreads();
  if (threadIdx.x < kChannelTile && valid) {
    for (int warp = 1; warp < kThreads / kWarpSize; ++warp) {
      mn = fminf(mn, minima[warp][c]);
      mx = fmaxf(mx, maxima[warp][c]);
    }
    const QuantParams p = quant_params(mn, mx, levels);
    aux_min[column] = p.minimum;
    tmp_max[column] = p.maximum;
    aux_scale[column] = p.scale;
    tmp_inv[column] = p.inverse;
    minima[0][c] = to_f32(p.minimum);
    maxima[0][c] = to_f32(p.inverse);
  }
  __syncthreads();
  if (valid) {
    const bf16 minimum = rn(minima[0][c]);
    const bf16 inverse = rn(maxima[0][c]);
    for (int t = token_lane; t < tokens; t += kThreads / kChannelTile) {
      const int64_t idx = (static_cast<int64_t>(t) * heads + h) * head_size + d;
      out[idx] = quant_value(src[idx], minimum, inverse, levels);
    }
  }
}

// A warp cooperates on one token, reading consecutive head/dimension values.
__global__ void quantize_token_kernel(
    const bf16* __restrict__ src, uint8_t* __restrict__ out,
    bf16* __restrict__ aux_min, bf16* __restrict__ aux_scale,
    bf16* __restrict__ tmp_max, bf16* __restrict__ tmp_inv, int tokens,
    int heads, int head_start, int head_count, int head_size, int levels) {
  const int t = blockIdx.x * kTokensPerBlock + threadIdx.x / kWarpSize;
  if (t >= tokens) return;
  const int lane = threadIdx.x % kWarpSize;
  const int columns = head_count * head_size;
  const int64_t row_base = static_cast<int64_t>(t) * heads * head_size;
  const int first = head_start * head_size;
  float mn = INFINITY;
  float mx = -INFINITY;
  for (int c = lane; c < columns; c += kWarpSize) {
    const float v = to_f32(src[row_base + first + c]);
    mn = fminf(mn, v);
    mx = fmaxf(mx, v);
  }
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    mn = fminf(mn, __shfl_xor_sync(0xffffffff, mn, offset));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, offset));
  }
  const QuantParams p = quant_params(mn, mx, levels);
  if (lane == 0) {
    aux_min[t] = p.minimum;
    tmp_max[t] = p.maximum;
    aux_scale[t] = p.scale;
    tmp_inv[t] = p.inverse;
  }
  for (int c = lane; c < columns; c += kWarpSize) {
    const int64_t idx = row_base + first + c;
    out[idx] = quant_value(src[idx], p.minimum, p.inverse, levels);
  }
}

__global__ void dequantize_channel_kernel(const uint8_t* __restrict__ u8,
                                          bf16* __restrict__ out,
                                          const bf16* __restrict__ aux_min,
                                          const bf16* __restrict__ aux_scale,
                                          int tokens, int heads, int head_start,
                                          int head_count, int head_size) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= head_count * head_size) return;
  const int h = head_start + column / head_size;
  const int d = column % head_size;
  const bf16 mn = aux_min[column];
  const bf16 scale = aux_scale[column];
  for (int t = 0; t < tokens; ++t) {
    const int64_t idx = (static_cast<int64_t>(t) * heads + h) * head_size + d;
    float v = static_cast<float>(u8[idx]);
    v = to_f32(rn(v * to_f32(scale)));
    out[idx] = rn(v + to_f32(mn));
  }
}

__global__ void dequantize_token_kernel(const uint8_t* __restrict__ u8,
                                        bf16* __restrict__ out,
                                        const bf16* __restrict__ aux_min,
                                        const bf16* __restrict__ aux_scale,
                                        int tokens, int heads, int head_start,
                                        int head_count, int head_size) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= tokens) return;
  const bf16 mn = aux_min[t];
  const bf16 scale = aux_scale[t];
  const int64_t row_base = static_cast<int64_t>(t) * heads * head_size;
  for (int h = head_start; h < head_start + head_count; ++h) {
    for (int d = 0; d < head_size; ++d) {
      const int64_t idx = row_base + static_cast<int64_t>(h) * head_size + d;
      float v = static_cast<float>(u8[idx]);
      v = to_f32(rn(v * to_f32(scale)));
      out[idx] = rn(v + to_f32(mn));
    }
  }
}

// Sign multiply + inverse head permutation + write back to the KV cache.
__global__ void scatter_cache_kernel(
    const bf16* __restrict__ src, const int64_t* __restrict__ head_order,
    const void* __restrict__ signs, const int64_t* __restrict__ block_indices,
    char* __restrict__ cache_base, int64_t cache_block_stride,
    int64_t cache_head_stride, int64_t cache_token_stride, int num_heads,
    int head_size, int kernel_tokens, int64_t total) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int d = static_cast<int>(idx % head_size);
  int64_t rest = idx / head_size;
  const int h = static_cast<int>(rest % num_heads);
  rest /= num_heads;
  const int t = static_cast<int>(rest % kernel_tokens);
  const int64_t p = rest / kernel_tokens;
  float value = to_f32(src[idx]);
  if (signs) {
    value *= to_f32(load_bf16(reinterpret_cast<const char*>(signs) +
                              (static_cast<int64_t>(h) * head_size + d) * 2));
  }
  const int64_t head = head_order ? head_order[h] : h;
  char* dst = cache_base + block_indices[p] * cache_block_stride +
              head * cache_head_stride + t * cache_token_stride +
              static_cast<int64_t>(d) * 2;
  *reinterpret_cast<bf16*>(dst) = rn(value);
}

__device__ __forceinline__ int tail_count_for_block(int tail_tokens,
                                                    int kernel_tokens,
                                                    int block) {
  const int remaining = tail_tokens - block * kernel_tokens;
  if (remaining <= 0) return 0;
  return remaining < kernel_tokens ? remaining : kernel_tokens;
}

__device__ __forceinline__ int64_t tail_block_offset(int tail_tokens,
                                                     int kernel_tokens,
                                                     int num_heads,
                                                     int head_size, int block) {
  int64_t offset = 0;
  for (int b = 0; b < block; ++b) {
    offset += static_cast<int64_t>(
                  tail_count_for_block(tail_tokens, kernel_tokens, b)) *
              num_heads * head_size * 2;
  }
  return offset;
}

// Encode: gather the trailing partial logical block from the cache.
__global__ void tail_pack_kernel(
    const char* __restrict__ cache_base, int64_t cache_block_stride,
    int64_t cache_head_stride, int64_t cache_token_stride,
    const int64_t* __restrict__ tail_indices, int num_heads, int head_size,
    int kernel_tokens, int tail_tokens, char* __restrict__ tail_out) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total =
      static_cast<int64_t>(tail_tokens) * num_heads * head_size;
  if (idx >= total) return;
  const int d = static_cast<int>(idx % head_size);
  int64_t rest = idx / head_size;
  const int h = static_cast<int>(rest % num_heads);
  const int t = static_cast<int>(rest / num_heads);
  const int block = t / kernel_tokens;
  const int in_block = t % kernel_tokens;
  const int count = tail_count_for_block(tail_tokens, kernel_tokens, block);
  if (in_block >= count) return;
  const char* src = cache_base + tail_indices[block] * cache_block_stride +
                    static_cast<int64_t>(h) * cache_head_stride +
                    static_cast<int64_t>(in_block) * cache_token_stride +
                    static_cast<int64_t>(d) * 2;
  const int64_t offset = tail_block_offset(tail_tokens, kernel_tokens,
                                           num_heads, head_size, block);
  const int64_t out_index =
      offset / 2 + (static_cast<int64_t>(h) * count + in_block) * head_size + d;
  char* dst = tail_out + out_index * 2;
  *reinterpret_cast<short*>(dst) = *reinterpret_cast<const short*>(src);
}

// Decode: scatter the trailing partial block back, zeroing unused tokens.
__global__ void tail_apply_kernel(
    const char* __restrict__ tail_in, const int64_t* __restrict__ tail_indices,
    char* __restrict__ cache_base, int64_t cache_block_stride,
    int64_t cache_head_stride, int64_t cache_token_stride, int num_heads,
    int head_size, int kernel_tokens, int tail_tokens) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total =
      static_cast<int64_t>(kernel_tokens) * num_heads * head_size;
  if (idx >= total) return;
  const int block = static_cast<int>(blockIdx.y);
  const int count = tail_count_for_block(tail_tokens, kernel_tokens, block);
  const int d = static_cast<int>(idx % head_size);
  int64_t rest = idx / head_size;
  const int h = static_cast<int>(rest % num_heads);
  const int in_block = static_cast<int>(rest / num_heads);
  char* dst = cache_base + tail_indices[block] * cache_block_stride +
              static_cast<int64_t>(h) * cache_head_stride +
              static_cast<int64_t>(in_block) * cache_token_stride +
              static_cast<int64_t>(d) * 2;
  if (in_block < count) {
    const int64_t offset = tail_block_offset(tail_tokens, kernel_tokens,
                                             num_heads, head_size, block);
    const int64_t in_index =
        offset / 2 + (static_cast<int64_t>(h) * count + in_block) * head_size +
        d;
    const char* src = tail_in + in_index * 2;
    *reinterpret_cast<short*>(dst) = *reinterpret_cast<const short*>(src);
  } else {
    *reinterpret_cast<short*>(dst) = 0;
  }
}

}  // namespace

namespace {

struct SectionDesc {
  int head_start;
  int head_count;
  int num_levels;
  int axis;
  int64_t aux_min_offset;
  int64_t aux_scale_offset;
};

struct PartDesc {
  const char* cache_base;
  int64_t block_stride;
  int64_t head_stride;
  int64_t token_stride;
  const void* signs;
  const int64_t* head_order;
  int num_heads;
  int head_size;
  int kernel_tokens;
  int64_t raw_offset;
  int64_t scratch_offset;
  int64_t u8_offset;
  int64_t raw_bytes;
  int uses_transformer;
  int quantized;
  int num_sections;
  const int64_t* sections;
  int block_size;
  int blocks_per_logical;
};

PartDesc load_part(const int64_t* row, const int64_t* section_base) {
  PartDesc p{};
  p.cache_base = reinterpret_cast<const char*>(row[KVS_PART_CACHE_BASE]);
  p.block_stride = row[KVS_PART_BLOCK_STRIDE];
  p.head_stride = row[KVS_PART_HEAD_STRIDE];
  p.token_stride = row[KVS_PART_TOKEN_STRIDE];
  p.signs = reinterpret_cast<const void*>(row[KVS_PART_SIGNS]);
  p.head_order = reinterpret_cast<const int64_t*>(row[KVS_PART_HEAD_ORDER]);
  p.num_heads = static_cast<int>(row[KVS_PART_NUM_HEADS]);
  p.head_size = static_cast<int>(row[KVS_PART_HEAD_SIZE]);
  p.kernel_tokens = static_cast<int>(row[KVS_PART_KERNEL_TOKENS]);
  p.raw_offset = row[KVS_PART_RAW_OFFSET];
  p.scratch_offset = row[KVS_PART_SCRATCH_OFFSET];
  p.u8_offset = row[KVS_PART_U8_OFFSET];
  p.raw_bytes = row[KVS_PART_RAW_BYTES];
  p.uses_transformer = static_cast<int>(row[KVS_PART_USES_TRANSFORMER]);
  p.quantized = static_cast<int>(row[KVS_PART_QUANTIZED]);
  p.num_sections = static_cast<int>(row[KVS_PART_NUM_SECTIONS]);
  p.sections = section_base + row[KVS_PART_SECTION_INDEX] * KVS_SECTION_FIELDS;
  p.block_size = static_cast<int>(row[KVS_PART_BLOCK_SIZE]);
  p.blocks_per_logical = static_cast<int>(row[KVS_PART_BLOCKS_PER_LOGICAL]);
  return p;
}

SectionDesc load_section(const int64_t* row) {
  SectionDesc s{};
  s.head_start = static_cast<int>(row[KVS_SECTION_HEAD_START]);
  s.head_count = static_cast<int>(row[KVS_SECTION_HEAD_COUNT]);
  s.num_levels = static_cast<int>(row[KVS_SECTION_NUM_LEVELS]);
  s.axis = static_cast<int>(row[KVS_SECTION_AXIS]);
  s.aux_min_offset = row[KVS_SECTION_AUX_MIN_OFFSET];
  s.aux_scale_offset = row[KVS_SECTION_AUX_SCALE_OFFSET];
  return s;
}

int64_t tail_bytes_for_part(const PartDesc& p, int tail_tokens) {
  int64_t bytes = 0;
  for (int b = 0; b < p.blocks_per_logical; ++b) {
    bytes +=
        static_cast<int64_t>(tail_count_host(tail_tokens, p.kernel_tokens, b)) *
        p.num_heads * p.head_size * 2;
  }
  return bytes;
}

// Gather + sign + fused normalized FWHT; result in canonical A.
void gather_and_transform(const PartDesc& p, const int64_t* indices,
                          int64_t body_blocks, char* canonical_a,
                          char* canonical_b, cudaStream_t stream) {
  const int64_t total =
      body_blocks * p.kernel_tokens * p.num_heads * p.head_size;
  const int blocks = blocks_for(total);
  gather_sign_kernel<<<blocks, kThreads, 0, stream>>>(
      p.cache_base, p.block_stride, p.head_stride, p.token_stride, indices,
      p.head_order, p.signs, p.num_heads, p.head_size, p.kernel_tokens, total,
      reinterpret_cast<bf16*>(canonical_b));
  check_launch("gather");
  if (!p.uses_transformer) {
    cudaMemcpyAsync(canonical_a, canonical_b, static_cast<size_t>(total) * 2,
                    cudaMemcpyDeviceToDevice, stream);
    check_launch("copy");
    return;
  }
  const int64_t rows = body_blocks * p.kernel_tokens * p.num_heads;
  launch_fwht(reinterpret_cast<const bf16*>(canonical_b),
              reinterpret_cast<bf16*>(canonical_a),
              reinterpret_cast<bf16*>(canonical_b), p.head_size, rows, stream);
}

void encode_part(const PartDesc& p, const int64_t* indices,
                 int64_t total_blocks, int64_t body_blocks, int valid_tokens,
                 char* arena_a, char* arena_b, char* aux, char* raw_tail,
                 int64_t u8_base, int64_t aux_half, int64_t* tail_cursor,
                 cudaStream_t stream) {
  char* canonical_a = arena_a + p.raw_offset;
  char* canonical_b = arena_a + p.scratch_offset;
  const int64_t total =
      body_blocks * p.kernel_tokens * p.num_heads * p.head_size;
  const int64_t body_tokens = body_blocks * p.kernel_tokens;

  if (body_blocks > 0) {
    gather_and_transform(p, indices, body_blocks, canonical_a, canonical_b,
                         stream);
    if (p.quantized) {
      uint8_t* u8 = reinterpret_cast<uint8_t*>(arena_b) + u8_base + p.u8_offset;
      for (int i = 0; i < p.num_sections; ++i) {
        const SectionDesc s = load_section(p.sections + i * KVS_SECTION_FIELDS);
        bf16* aux_min = reinterpret_cast<bf16*>(aux + s.aux_min_offset);
        bf16* aux_scale = reinterpret_cast<bf16*>(aux + s.aux_scale_offset);
        bf16* tmp_max =
            reinterpret_cast<bf16*>(aux + aux_half + s.aux_min_offset);
        bf16* tmp_inv =
            reinterpret_cast<bf16*>(aux + aux_half + s.aux_scale_offset);
        if (s.axis == KVS_AXIS_CHANNEL) {
          const int count = s.head_count * p.head_size;
          quantize_channel_kernel<<<ceil_div(count, kChannelTile), kThreads, 0,
                                    stream>>>(
              reinterpret_cast<const bf16*>(canonical_a), u8, aux_min,
              aux_scale, tmp_max, tmp_inv, static_cast<int>(body_tokens),
              p.num_heads, s.head_start, s.head_count, p.head_size,
              s.num_levels);
          check_launch("quantize_channel");
        } else {
          quantize_token_kernel<<<ceil_div(body_tokens, kTokensPerBlock), kThreads,
                                  0, stream>>>(
              reinterpret_cast<const bf16*>(canonical_a), u8, aux_min,
              aux_scale, tmp_max, tmp_inv, static_cast<int>(body_tokens),
              p.num_heads, s.head_start, s.head_count, p.head_size,
              s.num_levels);
          check_launch("quantize_token");
        }
      }
    } else {
      char* packed = arena_b + u8_base + p.u8_offset * 2;
      cudaMemcpyAsync(packed, canonical_a, static_cast<size_t>(total) * 2,
                      cudaMemcpyDeviceToDevice, stream);
      check_launch("pack");
    }
  }

  const int tail_tokens = valid_tokens % p.block_size;
  if (tail_tokens > 0 && raw_tail != nullptr) {
    const int64_t total_tail =
        static_cast<int64_t>(tail_tokens) * p.num_heads * p.head_size;
    tail_pack_kernel<<<blocks_for(total_tail), kThreads, 0, stream>>>(
        p.cache_base, p.block_stride, p.head_stride, p.token_stride,
        indices + total_blocks - p.blocks_per_logical, p.num_heads, p.head_size,
        p.kernel_tokens, tail_tokens, raw_tail + *tail_cursor);
    check_launch("tail_pack");
  }
  *tail_cursor += tail_bytes_for_part(p, tail_tokens);
}

void decode_part(const PartDesc& p, const int64_t* indices,
                 int64_t total_blocks, int64_t body_blocks, int valid_tokens,
                 char* arena_a, char* arena_b, char* aux, char* raw_tail,
                 int64_t u8_base, int64_t aux_half, int64_t* tail_cursor,
                 cudaStream_t stream) {
  char* canonical_a = arena_a + p.raw_offset;
  char* canonical_b = arena_a + p.scratch_offset;
  const int64_t total =
      body_blocks * p.kernel_tokens * p.num_heads * p.head_size;
  const int64_t body_tokens = body_blocks * p.kernel_tokens;

  if (body_blocks > 0) {
    cudaMemsetAsync(canonical_a, 0, static_cast<size_t>(total) * 2, stream);
    if (p.quantized) {
      const uint8_t* u8 =
          reinterpret_cast<const uint8_t*>(arena_b) + u8_base + p.u8_offset;
      for (int i = 0; i < p.num_sections; ++i) {
        const SectionDesc s = load_section(p.sections + i * KVS_SECTION_FIELDS);
        const bf16* aux_min =
            reinterpret_cast<const bf16*>(aux + s.aux_min_offset);
        const bf16* aux_scale =
            reinterpret_cast<const bf16*>(aux + s.aux_scale_offset);
        if (s.axis == KVS_AXIS_CHANNEL) {
          const int count = s.head_count * p.head_size;
          dequantize_channel_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
              u8, reinterpret_cast<bf16*>(canonical_a), aux_min, aux_scale,
              static_cast<int>(body_tokens), p.num_heads, s.head_start,
              s.head_count, p.head_size);
          check_launch("dequantize_channel");
        } else {
          dequantize_token_kernel<<<blocks_for(body_tokens), kThreads, 0,
                                    stream>>>(
              u8, reinterpret_cast<bf16*>(canonical_a), aux_min, aux_scale,
              static_cast<int>(body_tokens), p.num_heads, s.head_start,
              s.head_count, p.head_size);
          check_launch("dequantize_token");
        }
      }
    } else {
      const char* packed = arena_b + u8_base + p.u8_offset * 2;
      cudaMemcpyAsync(canonical_a, packed, static_cast<size_t>(total) * 2,
                      cudaMemcpyDeviceToDevice, stream);
      check_launch("unpack");
    }

    if (p.uses_transformer) {
      const int64_t rows = body_blocks * p.kernel_tokens * p.num_heads;
      launch_fwht(reinterpret_cast<const bf16*>(canonical_a),
                  reinterpret_cast<bf16*>(canonical_b),
                  reinterpret_cast<bf16*>(canonical_a), p.head_size, rows, stream);
    } else {
      cudaMemcpyAsync(canonical_b, canonical_a, static_cast<size_t>(total) * 2,
                      cudaMemcpyDeviceToDevice, stream);
      check_launch("icopy");
    }

    scatter_cache_kernel<<<blocks_for(total), kThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(canonical_b), p.head_order, p.signs,
        indices, const_cast<char*>(p.cache_base), p.block_stride, p.head_stride,
        p.token_stride, p.num_heads, p.head_size, p.kernel_tokens, total);
    check_launch("scatter");
  }

  const int tail_tokens = valid_tokens % p.block_size;
  if (tail_tokens > 0 && raw_tail != nullptr) {
    tail_apply_kernel<<<dim3(blocks_for(static_cast<int64_t>(p.kernel_tokens) *
                                        p.num_heads * p.head_size),
                             p.blocks_per_logical),
                        kThreads, 0, stream>>>(
        raw_tail + *tail_cursor, indices + total_blocks - p.blocks_per_logical,
        const_cast<char*>(p.cache_base), p.block_stride, p.head_stride,
        p.token_stride, p.num_heads, p.head_size, p.kernel_tokens, tail_tokens);
    check_launch("tail_apply");
  }
  *tail_cursor += tail_bytes_for_part(p, tail_tokens);
}

}  // namespace

extern "C" void kvs_hadamard_flat(int64_t input_ptr, int64_t output_ptr,
                                    int64_t rows, int head_size,
                                    int64_t stream_ptr) {
  launch_fwht(reinterpret_cast<const bf16*>(input_ptr),
              reinterpret_cast<bf16*>(output_ptr),
              reinterpret_cast<bf16*>(input_ptr), head_size, rows,
              reinterpret_cast<cudaStream_t>(stream_ptr));
}

extern "C" int kvs_encode_flat(int64_t meta_ptr, int64_t indices_ptr,
                               int64_t num_blocks, int64_t valid_tokens,
                               int64_t stream_ptr) {
  const int64_t* meta = reinterpret_cast<const int64_t*>(meta_ptr);
  const int64_t num_parts = meta[KVS_META_NUM_PARTS];
  if (num_parts <= 0) return -1;
  char* arena_a = reinterpret_cast<char*>(meta[KVS_META_ARENA_A]);
  char* arena_b = reinterpret_cast<char*>(meta[KVS_META_ARENA_B]);
  char* aux = reinterpret_cast<char*>(meta[KVS_META_AUX]);
  char* raw_tail = reinterpret_cast<char*>(meta[KVS_META_RAW_TAIL]);
  const int64_t u8_base = meta[KVS_META_U8_BASE];
  const int64_t aux_half = meta[KVS_META_AUX_ARENA_BYTES] / 2;
  const int64_t* part_rows = meta + KVS_META_HEADER_FIELDS;
  const int64_t* section_rows = part_rows + num_parts * KVS_PART_FIELDS;
  const int64_t* indices = reinterpret_cast<const int64_t*>(indices_ptr);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  const PartDesc first = load_part(part_rows, section_rows);
  const int64_t body_blocks =
      (valid_tokens / first.block_size) * first.blocks_per_logical;
  (void)num_blocks;

  int64_t tail_cursor = 0;
  for (int64_t i = 0; i < num_parts; ++i) {
    const PartDesc p = load_part(part_rows + i * KVS_PART_FIELDS, section_rows);
    encode_part(p, indices, num_blocks, body_blocks,
                static_cast<int>(valid_tokens), arena_a, arena_b, aux, raw_tail,
                u8_base, aux_half, &tail_cursor, stream);
  }
  return 0;
}

extern "C" int kvs_decode_flat(int64_t meta_ptr, int64_t indices_ptr,
                               int64_t num_blocks, int64_t valid_tokens,
                               int64_t u8_bytes, int64_t stream_ptr) {
  const int64_t* meta = reinterpret_cast<const int64_t*>(meta_ptr);
  const int64_t num_parts = meta[KVS_META_NUM_PARTS];
  if (num_parts <= 0) return -1;
  char* arena_a = reinterpret_cast<char*>(meta[KVS_META_ARENA_A]);
  char* arena_b = reinterpret_cast<char*>(meta[KVS_META_ARENA_B]);
  char* aux = reinterpret_cast<char*>(meta[KVS_META_AUX]);
  char* raw_tail = reinterpret_cast<char*>(meta[KVS_META_RAW_TAIL]);
  const int64_t u8_base = meta[KVS_META_U8_BASE];
  const int64_t aux_half = meta[KVS_META_AUX_ARENA_BYTES] / 2;
  const int64_t* part_rows = meta + KVS_META_HEADER_FIELDS;
  const int64_t* section_rows = part_rows + num_parts * KVS_PART_FIELDS;
  const int64_t* indices = reinterpret_cast<const int64_t*>(indices_ptr);
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

  const PartDesc first = load_part(part_rows, section_rows);
  const int64_t body_blocks =
      (valid_tokens / first.block_size) * first.blocks_per_logical;
  (void)num_blocks;
  (void)u8_bytes;

  int64_t tail_cursor = 0;
  for (int64_t i = 0; i < num_parts; ++i) {
    const PartDesc p = load_part(part_rows + i * KVS_PART_FIELDS, section_rows);
    decode_part(p, indices, num_blocks, body_blocks,
                static_cast<int>(valid_tokens), arena_a, arena_b, aux, raw_tail,
                u8_base, aux_half, &tail_cursor, stream);
  }
  return 0;
}
