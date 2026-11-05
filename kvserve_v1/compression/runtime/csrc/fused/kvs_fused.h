// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Fast capability: uniform BF16 HND block-1 parts, D128, a single section
// spanning all heads, K/channel and V/token quantization with Hadamard. Heads
// may be 1/2/4/8/16. Layer count and quantization levels come from metadata.
//
// The caller owns metadata, pinned host IDs, device IDs, arenas and stream,
// selects their CUDA device, and keeps all owners alive until stream completion.
// No allocation, registration, synchronization or default-stream work occurs
// in either serving entry point. The decode input is already ANS-decompressed
// in arena B. Both entry points use compact ceil(tokens/16)*16 layer strides.
// Packed arenas must therefore fit the padded capacity, including an odd full
// capacity if a direct ABI caller uses one. Python capability may be stricter.
extern "C" {
void kvs_fused_bind_shared(int device);
void kvs_fused_encode(int64_t meta_cpu, int64_t meta_gpu, int64_t host_ids,
                      int64_t gpu_ids, int tokens, int part, int capacity,
                      int heads, int64_t stream);
void kvs_fused_decode(int64_t meta_cpu, int64_t meta_gpu, int64_t host_ids,
                      int64_t gpu_ids, int tokens, int part, int capacity,
                      int heads, int64_t stream);
}
