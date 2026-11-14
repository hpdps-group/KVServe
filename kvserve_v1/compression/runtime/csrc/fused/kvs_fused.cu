// SPDX-License-Identifier: Apache-2.0
// Framework-neutral paged gather/FWHT/quantize and restore kernels. Include the
// existing backend once to share its metadata ABI, quant helpers and generic
// flat entry points; do not fork a copy of the baseline implementation.
#include "../kvs_compress.cu"
#include "kvs_fused.h"

#include <algorithm>
#include <limits>

namespace {

constexpr int kFusedHeadSize = 128;
constexpr int kFusedHeadThreads = 1024;
constexpr int kFusedHeadWarps = kFusedHeadThreads / kWarpSize;
constexpr int kRegisterTailTokens = 128;
constexpr int kSharedTokenLimit = 1024 - kRegisterTailTokens;
constexpr int kSharedStatisticsBytes = kFusedHeadSize * 2 * sizeof(unsigned);
constexpr int kRequiredSharedBytes =
    kSharedTokenLimit * kFusedHeadSize * sizeof(bf16) + kSharedStatisticsBytes;
constexpr int kValueWarps = 4;
constexpr int kRestoreWarps = 8;

void fused_checked(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("KVServe fused ") + operation + ": " +
                             cudaGetErrorString(status));
  }
}

// Core load_part/load_section are host helpers. Only the small descriptor load
// is device-specific here; no core source or global preprocessor qualifier is
// changed. All offsets retain the core ABI constants.
__device__ __forceinline__ PartDesc fused_part(const int64_t* meta) {
  const int64_t* row =
      meta + KVS_META_HEADER_FIELDS + blockIdx.y * KVS_PART_FIELDS;
  const int64_t* sections = meta + KVS_META_HEADER_FIELDS +
                            meta[KVS_META_NUM_PARTS] * KVS_PART_FIELDS;
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
  p.sections = sections + row[KVS_PART_SECTION_INDEX] * KVS_SECTION_FIELDS;
  return p;
}

__device__ __forceinline__ SectionDesc fused_section(const PartDesc& p) {
  SectionDesc s{};
  s.head_start = static_cast<int>(p.sections[KVS_SECTION_HEAD_START]);
  s.head_count = static_cast<int>(p.sections[KVS_SECTION_HEAD_COUNT]);
  s.num_levels = static_cast<int>(p.sections[KVS_SECTION_NUM_LEVELS]);
  s.axis = static_cast<int>(p.sections[KVS_SECTION_AXIS]);
  s.aux_min_offset = p.sections[KVS_SECTION_AUX_MIN_OFFSET];
  s.aux_scale_offset = p.sections[KVS_SECTION_AUX_SCALE_OFFSET];
  return s;
}

// A lane owns dimensions lane + {0,32,64,96}. Round after all seven butterfly
// stages and again after normalization, exactly as the existing BF16 backend.
__device__ __forceinline__ void fused_fwht(float (&value)[4]) {
  const int lane = threadIdx.x % kWarpSize;
  __nv_bfloat162 pair[2] = {
      __halves2bfloat162(rn(value[0]), rn(value[1])),
      __halves2bfloat162(rn(value[2]), rn(value[3]))};
#pragma unroll
  for (int step = 1; step < kWarpSize; step <<= 1) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      unsigned bits = *reinterpret_cast<const unsigned*>(&pair[i]);
      bits = __shfl_xor_sync(0xffffffff, bits, step);
      const auto other = *reinterpret_cast<const __nv_bfloat162*>(&bits);
      pair[i] = (lane & step) ? __hsub2(other, pair[i]) : __hadd2(pair[i], other);
    }
  }
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const bf16 a = __low2bfloat16(pair[i]);
    const bf16 b = __high2bfloat16(pair[i]);
    pair[i] = __halves2bfloat162(__hadd(a, b), __hsub(a, b));
  }
  const auto a = pair[0];
  const auto b = pair[1];
  pair[0] = __hadd2(a, b);
  pair[1] = __hsub2(a, b);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    value[2 * i] =
        to_f32(rn(to_f32(__low2bfloat16(pair[i])) * 0.08838834764831845f));
    value[2 * i + 1] =
        to_f32(rn(to_f32(__high2bfloat16(pair[i])) * 0.08838834764831845f));
  }
}

__device__ __forceinline__ unsigned fused_ordered(float value) {
  const unsigned bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ float fused_unordered(unsigned value) {
  return __uint_as_float((value & 0x80000000u) ? (value ^ 0x80000000u) : ~value);
}

__device__ __forceinline__ void fused_key_values(
    const PartDesc& p, const int64_t* ids, int token, int head,
    const float (&signs)[4], float (&value)[4]) {
  const int lane = threadIdx.x % kWarpSize;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int dimension = i * kWarpSize + lane;
    const char* address = p.cache_base + ids[token] * p.block_stride +
                          head * p.head_stride + dimension * sizeof(bf16);
    value[i] = to_f32(rn(to_f32(load_bf16(address)) * signs[i]));
  }
  fused_fwht(value);
}

// One block per logical head/layer. Values remain in shared memory/registers
// between the complete-token channel statistics and quantization pass.
__global__ __launch_bounds__(kFusedHeadThreads) void fused_key_encode(
    const int64_t* meta, const int64_t* ids, int tokens, int shared_tokens,
    int compact_tokens) {
  constexpr int tail_rows = kRegisterTailTokens / kFusedHeadWarps;
  extern __shared__ unsigned char storage[];
  auto* raw = reinterpret_cast<bf16*>(storage);
  auto* minima = reinterpret_cast<unsigned*>(
      storage + shared_tokens * kFusedHeadSize * sizeof(bf16));
  auto* maxima = minima + kFusedHeadSize;
  const PartDesc p = fused_part(meta);
  const SectionDesc section = fused_section(p);
  const int logical_head = blockIdx.x;
  const int source_head = p.head_order ? p.head_order[logical_head] : logical_head;
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  float signs[4], lo[4], hi[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    signs[i] = to_f32(static_cast<const bf16*>(p.signs)[
        logical_head * kFusedHeadSize + i * kWarpSize + lane]);
    lo[i] = INFINITY;
    hi[i] = -INFINITY;
  }
  if (threadIdx.x < kFusedHeadSize) {
    minima[threadIdx.x] = fused_ordered(INFINITY);
    maxima[threadIdx.x] = fused_ordered(-INFINITY);
  }
  unsigned tail[tail_rows][2];
#pragma unroll 1
  for (int token = warp; token < shared_tokens; token += kFusedHeadWarps) {
    float value[4];
    fused_key_values(p, ids, token, source_head, signs, value);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const bf16 rounded = rn(value[i]);
      raw[token * kFusedHeadSize + i * kWarpSize + lane] = rounded;
      lo[i] = fminf(lo[i], to_f32(rounded));
      hi[i] = fmaxf(hi[i], to_f32(rounded));
    }
  }
#pragma unroll
  for (int j = 0; j < tail_rows; ++j) {
    const int token = shared_tokens + warp + j * kFusedHeadWarps;
    if (token < tokens) {
      float value[4];
      fused_key_values(p, ids, token, source_head, signs, value);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const bf16 a = rn(value[2 * i]);
        const bf16 b = rn(value[2 * i + 1]);
        const auto pair = __halves2bfloat162(a, b);
        tail[j][i] = *reinterpret_cast<const unsigned*>(&pair);
        lo[2 * i] = fminf(lo[2 * i], to_f32(a));
        hi[2 * i] = fmaxf(hi[2 * i], to_f32(a));
        lo[2 * i + 1] = fminf(lo[2 * i + 1], to_f32(b));
        hi[2 * i + 1] = fmaxf(hi[2 * i + 1], to_f32(b));
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int column = i * kWarpSize + lane;
    atomicMin(minima + column, fused_ordered(lo[i]));
    atomicMax(maxima + column, fused_ordered(hi[i]));
  }
  __syncthreads();
  char* aux = reinterpret_cast<char*>(meta[KVS_META_AUX]);
  auto* aux_min = reinterpret_cast<bf16*>(aux + section.aux_min_offset);
  auto* aux_scale = reinterpret_cast<bf16*>(aux + section.aux_scale_offset);
  auto* tmp_max = reinterpret_cast<bf16*>(
      aux + meta[KVS_META_AUX_ARENA_BYTES] / 2 + section.aux_min_offset);
  auto* tmp_inv = reinterpret_cast<bf16*>(
      aux + meta[KVS_META_AUX_ARENA_BYTES] / 2 + section.aux_scale_offset);
  if (threadIdx.x < kFusedHeadSize) {
    const int dimension = threadIdx.x;
    const int column = logical_head * kFusedHeadSize + dimension;
    const QuantParams q = quant_params(fused_unordered(minima[dimension]),
                                        fused_unordered(maxima[dimension]),
                                        section.num_levels);
    aux_min[column] = q.minimum;
    aux_scale[column] = q.scale;
    tmp_max[column] = q.maximum;
    tmp_inv[column] = q.inverse;
    minima[dimension] = __float_as_uint(to_f32(q.minimum));
    maxima[dimension] = __float_as_uint(to_f32(q.inverse));
  }
  __syncthreads();
  bf16 minimum[4], inverse[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    minimum[i] = rn(__uint_as_float(minima[i * kWarpSize + lane]));
    inverse[i] = rn(__uint_as_float(maxima[i * kWarpSize + lane]));
  }
  const int row_elements = p.num_heads * kFusedHeadSize;
  auto* output = reinterpret_cast<uint8_t*>(
      meta[KVS_META_ARENA_B] + meta[KVS_META_U8_BASE] +
      static_cast<int64_t>(blockIdx.y) * compact_tokens * row_elements);
#pragma unroll 1
  for (int token = warp; token < shared_tokens; token += kFusedHeadWarps) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int column = logical_head * kFusedHeadSize + i * kWarpSize + lane;
      output[static_cast<int64_t>(token) * row_elements + column] = quant_value(
          raw[token * kFusedHeadSize + i * kWarpSize + lane],
          minimum[i], inverse[i], section.num_levels);
    }
  }
#pragma unroll
  for (int j = 0; j < tail_rows; ++j) {
    const int token = shared_tokens + warp + j * kFusedHeadWarps;
    if (token < tokens) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const unsigned bits = tail[j][i];
        const auto pair = *reinterpret_cast<const __nv_bfloat162*>(&bits);
        const int column = logical_head * kFusedHeadSize + 2 * i * kWarpSize + lane;
        output[static_cast<int64_t>(token) * row_elements + column] = quant_value(
            __low2bfloat16(pair), minimum[2 * i], inverse[2 * i], section.num_levels);
        output[static_cast<int64_t>(token) * row_elements + column + kWarpSize] =
            quant_value(__high2bfloat16(pair), minimum[2 * i + 1],
                        inverse[2 * i + 1], section.num_levels);
      }
    }
  }
}

template <int Heads>
__global__ void fused_value_encode(const int64_t* meta, const int64_t* ids,
                                    int tokens, int compact_tokens) {
  const PartDesc p = fused_part(meta);
  const SectionDesc section = fused_section(p);
  const int lane = threadIdx.x % kWarpSize;
  const int token = blockIdx.x * kValueWarps + threadIdx.x / kWarpSize;
  __nv_bfloat162 value[Heads * 2];
#pragma unroll
  for (int i = 0; i < Heads * 2; ++i) {
    const int logical_head = i / 2;
    const int dimension = (i % 2) * 64 + lane * 2;
    auto pair = __halves2bfloat162(rn(0.f), rn(0.f));
    if (token < tokens) {
      const int source_head = p.head_order ? p.head_order[logical_head] : logical_head;
      const char* source = p.cache_base + ids[token] * p.block_stride +
                           source_head * p.head_stride + dimension * sizeof(bf16);
      const auto input = *reinterpret_cast<const __nv_bfloat162*>(source);
      const auto signs = *reinterpret_cast<const __nv_bfloat162*>(
          static_cast<const bf16*>(p.signs) + logical_head * kFusedHeadSize + dimension);
      pair = __halves2bfloat162(
          rn(to_f32(__low2bfloat16(input)) * to_f32(__low2bfloat16(signs))),
          rn(to_f32(__high2bfloat16(input)) * to_f32(__high2bfloat16(signs))));
    }
    const bf16 a = __low2bfloat16(pair);
    const bf16 b = __high2bfloat16(pair);
    value[i] = __halves2bfloat162(__hadd(a, b), __hsub(a, b));
  }
#pragma unroll
  for (int step = 1; step < kWarpSize; step <<= 1) {
#pragma unroll
    for (int i = 0; i < Heads * 2; ++i) {
      unsigned bits = *reinterpret_cast<const unsigned*>(&value[i]);
      bits = __shfl_xor_sync(0xffffffff, bits, step);
      const auto other = *reinterpret_cast<const __nv_bfloat162*>(&bits);
      value[i] = (lane & step) ? __hsub2(other, value[i]) : __hadd2(value[i], other);
    }
  }
#pragma unroll
  for (int h = 0; h < Heads; ++h) {
    const auto a = value[h * 2];
    const auto b = value[h * 2 + 1];
    value[h * 2] = __hadd2(a, b);
    value[h * 2 + 1] = __hsub2(a, b);
  }
  float lo = INFINITY;
  float hi = -INFINITY;
#pragma unroll
  for (int i = 0; i < Heads * 2; ++i) {
    const bf16 a = rn(to_f32(__low2bfloat16(value[i])) * 0.08838834764831845f);
    const bf16 b = rn(to_f32(__high2bfloat16(value[i])) * 0.08838834764831845f);
    value[i] = __halves2bfloat162(a, b);
    lo = fminf(lo, to_f32(a));
    lo = fminf(lo, to_f32(b));
    hi = fmaxf(hi, to_f32(a));
    hi = fmaxf(hi, to_f32(b));
  }
#pragma unroll
  for (int step = kWarpSize / 2; step; step >>= 1) {
    lo = fminf(lo, __shfl_xor_sync(0xffffffff, lo, step));
    hi = fmaxf(hi, __shfl_xor_sync(0xffffffff, hi, step));
  }
  if (token < tokens) {
    const QuantParams q = quant_params(lo, hi, section.num_levels);
    char* aux = reinterpret_cast<char*>(meta[KVS_META_AUX]);
    auto* minimum = reinterpret_cast<bf16*>(aux + section.aux_min_offset);
    auto* scale = reinterpret_cast<bf16*>(aux + section.aux_scale_offset);
    auto* maximum = reinterpret_cast<bf16*>(
        aux + meta[KVS_META_AUX_ARENA_BYTES] / 2 + section.aux_min_offset);
    auto* inverse = reinterpret_cast<bf16*>(
        aux + meta[KVS_META_AUX_ARENA_BYTES] / 2 + section.aux_scale_offset);
    if (lane == 0) {
      minimum[token] = q.minimum;
      scale[token] = q.scale;
      maximum[token] = q.maximum;
      inverse[token] = q.inverse;
    }
    constexpr int row_elements = Heads * kFusedHeadSize;
    auto* output = reinterpret_cast<uint8_t*>(
        meta[KVS_META_ARENA_B] + meta[KVS_META_U8_BASE] +
        static_cast<int64_t>(blockIdx.y) * compact_tokens * row_elements);
#pragma unroll
    for (int i = 0; i < Heads * 2; ++i) {
      const int column = (i / 2) * kFusedHeadSize + (i % 2) * 64 + lane * 2;
      const uint16_t a = quant_value(__low2bfloat16(value[i]), q.minimum,
                                      q.inverse, section.num_levels);
      const uint16_t b = quant_value(__high2bfloat16(value[i]), q.minimum,
                                      q.inverse, section.num_levels);
      *reinterpret_cast<uint16_t*>(output + static_cast<int64_t>(token) * row_elements + column) =
          a | (b << 8);
    }
  }
}

__global__ void fused_restore(const int64_t* meta, const int64_t* ids,
                                int tokens, int part, int compact_tokens) {
  const PartDesc p = fused_part(meta);
  const SectionDesc section = fused_section(p);
  const int row = blockIdx.x * kRestoreWarps + threadIdx.x / kWarpSize;
  const int token = row / p.num_heads;
  const int logical_head = row % p.num_heads;
  const int lane = threadIdx.x % kWarpSize;
  const int row_elements = p.num_heads * kFusedHeadSize;
  const auto* input = reinterpret_cast<const uint8_t*>(
      meta[KVS_META_ARENA_B] + meta[KVS_META_U8_BASE] +
      static_cast<int64_t>(blockIdx.y) * compact_tokens * row_elements);
  const char* aux = reinterpret_cast<const char*>(meta[KVS_META_AUX]);
  const auto* minimum = reinterpret_cast<const bf16*>(aux + section.aux_min_offset);
  const auto* scale = reinterpret_cast<const bf16*>(aux + section.aux_scale_offset);
  float value[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int dimension = i * kWarpSize + lane;
    value[i] = 0.f;
    if (token < tokens) {
      const int column = logical_head * kFusedHeadSize + dimension;
      const int qindex = part == 0 ? column : token;
      const float symbol = input[static_cast<int64_t>(token) * row_elements + column];
      const bf16 product = rn(symbol * to_f32(scale[qindex]));
      value[i] = to_f32(rn(to_f32(product) + to_f32(minimum[qindex])));
    }
  }
  fused_fwht(value);
  if (token < tokens) {
    const int target_head = p.head_order ? p.head_order[logical_head] : logical_head;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dimension = i * kWarpSize + lane;
      const float sign = to_f32(static_cast<const bf16*>(p.signs)[
          logical_head * kFusedHeadSize + dimension]);
      char* output = const_cast<char*>(p.cache_base) + ids[token] * p.block_stride +
                     target_head * p.head_stride + dimension * sizeof(bf16);
      *reinterpret_cast<bf16*>(output) = rn(value[i] * sign);
    }
  }
}

int validate_fused(int64_t meta_cpu, int64_t meta_gpu, int64_t host_ids,
                    int64_t gpu_ids, int tokens, int part, int capacity,
                    int heads, int64_t stream) {
  if (!meta_cpu || !meta_gpu || !host_ids || !gpu_ids || !stream ||
      capacity < 1 || capacity > 1024 || tokens < 1 || tokens > capacity ||
      part < 0 || part > 1 ||
      !(heads == 1 || heads == 2 || heads == 4 || heads == 8 || heads == 16)) {
    throw std::invalid_argument("Invalid fused frame pointers, geometry or explicit stream");
  }
  const auto* meta = reinterpret_cast<const int64_t*>(meta_cpu);
  const int64_t layers = meta[KVS_META_NUM_PARTS];
  if (layers < 1 || layers > 65535 || !meta[KVS_META_ARENA_A] ||
      !meta[KVS_META_ARENA_B] || !meta[KVS_META_AUX] ||
      meta[KVS_META_U8_BASE] < 0 || meta[KVS_META_U8_BASE] % 2 ||
      meta[KVS_META_AUX_ARENA_BYTES] <= 0 || meta[KVS_META_AUX_ARENA_BYTES] % 2) {
    throw std::invalid_argument("Invalid fused metadata header or CUDA grid extent");
  }
  const int64_t* parts = meta + KVS_META_HEADER_FIELDS;
  const int64_t* sections = parts + layers * KVS_PART_FIELDS;
  const int expected_axis = part == 0 ? KVS_AXIS_CHANNEL : KVS_AXIS_TOKEN;
  const int64_t raw_bytes = static_cast<int64_t>(capacity) * heads * kFusedHeadSize * sizeof(bf16);
  const int64_t aux_capacity = meta[KVS_META_AUX_ARENA_BYTES] / 2;
  const int64_t auxiliary_elements = part == 0 ? heads * kFusedHeadSize : capacity;
  for (int64_t layer = 0; layer < layers; ++layer) {
    const int64_t* row = parts + layer * KVS_PART_FIELDS;
    const int64_t section_index = row[KVS_PART_SECTION_INDEX];
    if (row[KVS_PART_NUM_HEADS] != heads || row[KVS_PART_HEAD_SIZE] != kFusedHeadSize ||
        row[KVS_PART_KERNEL_TOKENS] != 1 || row[KVS_PART_BLOCK_SIZE] != 1 ||
        row[KVS_PART_BLOCKS_PER_LOGICAL] != 1 || row[KVS_PART_USES_TRANSFORMER] != 1 ||
        row[KVS_PART_QUANTIZED] != 1 || row[KVS_PART_NUM_SECTIONS] != 1 ||
        row[KVS_PART_RAW_BYTES] != raw_bytes || !row[KVS_PART_SIGNS] ||
        !row[KVS_PART_CACHE_BASE] || row[KVS_PART_BLOCK_STRIDE] <= 0 ||
        row[KVS_PART_HEAD_STRIDE] <= 0 || section_index < 0 || section_index >= layers) {
      throw std::invalid_argument("Fused part is not uniform BF16 block-1 Hadamard D128");
    }
    if (part == 1 && (row[KVS_PART_CACHE_BASE] % 4 ||
                      row[KVS_PART_BLOCK_STRIDE] % 4 ||
                      (heads > 1 && row[KVS_PART_HEAD_STRIDE] % 4))) {
      throw std::invalid_argument("BF16x2 V needs aligned base and active strides");
    }
    const int64_t* section = sections + section_index * KVS_SECTION_FIELDS;
    if (section[KVS_SECTION_HEAD_START] != 0 || section[KVS_SECTION_HEAD_COUNT] != heads ||
        section[KVS_SECTION_AXIS] != expected_axis || section[KVS_SECTION_NUM_LEVELS] < 2 ||
        section[KVS_SECTION_NUM_LEVELS] > 256 || section[KVS_SECTION_AUX_MIN_OFFSET] < 0 ||
        section[KVS_SECTION_AUX_SCALE_OFFSET] < 0 ||
        section[KVS_SECTION_AUX_MIN_OFFSET] % 2 || section[KVS_SECTION_AUX_SCALE_OFFSET] % 2 ||
        section[KVS_SECTION_AUX_MIN_OFFSET] + auxiliary_elements * 2 > aux_capacity ||
        section[KVS_SECTION_AUX_SCALE_OFFSET] + auxiliary_elements * 2 > aux_capacity) {
      throw std::invalid_argument("Fused section does not cover the complete part/axis");
    }
  }
  return static_cast<int>(layers);
}

void upload_indices(int64_t host_ids, int64_t gpu_ids, int tokens, cudaStream_t stream) {
  fused_checked(cudaMemcpyAsync(reinterpret_cast<void*>(gpu_ids),
                                reinterpret_cast<const void*>(host_ids),
                                tokens * sizeof(int64_t), cudaMemcpyHostToDevice, stream),
                "upload indices");
}

}  // namespace

extern "C" void kvs_fused_bind_shared(int device) {
  int original = 0;
  fused_checked(cudaGetDevice(&original), "get current device");
  fused_checked(cudaSetDevice(device), "select bind device");
  try {
    int shared = 0;
    fused_checked(cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                         device), "query shared-memory capability");
    if (shared < kRequiredSharedBytes) {
      throw std::runtime_error("Fused K requires 225 KiB opt-in shared memory per block");
    }
    fused_checked(cudaFuncSetAttribute(fused_key_encode,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       kRequiredSharedBytes), "opt-in K shared memory");
    fused_checked(cudaSetDevice(original), "restore bind device");
  } catch (...) {
    cudaSetDevice(original);
    throw;
  }
}

extern "C" void kvs_fused_encode(int64_t meta_cpu, int64_t meta_gpu,
                                   int64_t host_ids, int64_t gpu_ids, int tokens,
                                   int part, int capacity, int heads, int64_t stream) {
  const int layers = validate_fused(meta_cpu, meta_gpu, host_ids, gpu_ids,
                                     tokens, part, capacity, heads, stream);
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  upload_indices(host_ids, gpu_ids, tokens, cuda_stream);
  const int compact_tokens = (tokens + 15) / 16 * 16;
  const int row_elements = heads * kFusedHeadSize;
  const auto* host = reinterpret_cast<const int64_t*>(meta_cpu);
  // Short V capacities have 256-byte-aligned auxiliary gaps even for a full frame.
  if (tokens != capacity || compact_tokens != tokens || (part == 1 && capacity % 128 != 0)) {
    fused_checked(cudaMemsetAsync(
        reinterpret_cast<void*>(host[KVS_META_ARENA_B] + host[KVS_META_U8_BASE]), 0,
        static_cast<size_t>(layers) * compact_tokens * row_elements, cuda_stream),
        "clear partial packed frame");
    fused_checked(cudaMemsetAsync(reinterpret_cast<void*>(host[KVS_META_AUX]), 0,
                                  host[KVS_META_AUX_ARENA_BYTES] / 2, cuda_stream),
                  "clear partial auxiliary frame");
  }
  const auto* gpu = reinterpret_cast<const int64_t*>(meta_gpu);
  const auto* indices = reinterpret_cast<const int64_t*>(gpu_ids);
  if (part == 0) {
    const int shared_tokens = std::min(tokens, kSharedTokenLimit);
    const size_t shared_bytes = shared_tokens * kFusedHeadSize * sizeof(bf16) +
                                kSharedStatisticsBytes;
    fused_key_encode<<<dim3(heads, layers), kFusedHeadThreads, shared_bytes, cuda_stream>>>(
        gpu, indices, tokens, shared_tokens, compact_tokens);
    check_launch("fused key gather/FWHT/statistics/quantize");
    return;
  }
#define KVS_VALUE_HEAD_CASE(H)                                                \
  case H:                                                                    \
    fused_value_encode<H><<<dim3(ceil_div(tokens, kValueWarps), layers),        \
                             kValueWarps * kWarpSize, 0, cuda_stream>>>(       \
        gpu, indices, tokens, compact_tokens);                                \
    break
  switch (heads) {
    KVS_VALUE_HEAD_CASE(1);
    KVS_VALUE_HEAD_CASE(2);
    KVS_VALUE_HEAD_CASE(4);
    KVS_VALUE_HEAD_CASE(8);
    KVS_VALUE_HEAD_CASE(16);
  }
#undef KVS_VALUE_HEAD_CASE
  check_launch("fused vector V gather/FWHT/statistics/quantize");
}

extern "C" void kvs_fused_decode(int64_t meta_cpu, int64_t meta_gpu,
                                   int64_t host_ids, int64_t gpu_ids, int tokens,
                                   int part, int capacity, int heads, int64_t stream,
                                   bool standard_layout) {
  const int layers = validate_fused(meta_cpu, meta_gpu, host_ids, gpu_ids,
                                     tokens, part, capacity, heads, stream);
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  upload_indices(host_ids, gpu_ids, tokens, cuda_stream);
  const int compact_tokens = standard_layout ? capacity : (tokens + 15) / 16 * 16;
  fused_restore<<<dim3(ceil_div(static_cast<int64_t>(tokens) * heads, kRestoreWarps), layers),
                    kRestoreWarps * kWarpSize, 0, cuda_stream>>>(
      reinterpret_cast<const int64_t*>(meta_gpu),
      reinterpret_cast<const int64_t*>(gpu_ids), tokens, part, compact_tokens);
  check_launch("fused dequantize/inverse-FWHT/scatter");
}
