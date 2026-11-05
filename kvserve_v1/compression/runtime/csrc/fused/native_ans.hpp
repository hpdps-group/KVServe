// SPDX-License-Identifier: Apache-2.0
// Default nvCOMP ANS manager promoted from the verified integration prototype.
#pragma once
// Opt-in nvCOMP C++ ANSManager adapter. No Python owners or stream callbacks
// are created on the compression/decompression submission path.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime_api.h>
#include <nvcomp/ans.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace kvserve_fused_native {

static void checked(cudaError_t rc) {
  if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}

class FixedBank {
 public:
  FixedBank(uintptr_t pointer, size_t capacity)
      : base_(reinterpret_cast<uint8_t*>(pointer)), capacity_(capacity) {
    free_.emplace(0, capacity);
  }

  void* allocate(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t aligned = (bytes + 255) & ~size_t(255);
    auto best = free_.end();
    for (auto it = free_.begin(); it != free_.end(); ++it) {
      if (it->second >= aligned &&
          (best == free_.end() || it->second < best->second)) best = it;
    }
    if (best == free_.end())
      throw std::runtime_error("Native ANS fixed workspace exhausted: request=" +
                               std::to_string(aligned) + " capacity=" +
                               std::to_string(capacity_));
    const size_t offset = best->first, remaining = best->second - aligned;
    free_.erase(best);
    if (remaining) free_.emplace(offset + aligned, remaining);
    live_.emplace(offset, aligned);
    live_bytes_ += aligned;
    peak_bytes_ = std::max(peak_bytes_, live_bytes_);
    return base_ + offset;
  }

  void release(void* pointer, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t offset = static_cast<uint8_t*>(pointer) - base_;
    auto live = live_.find(offset);
    if (live == live_.end())
      throw std::runtime_error("Native ANS invalid fixed workspace release");
    const size_t length = live->second;
    live_bytes_ -= length;
    live_.erase(live);
    auto current = free_.emplace(offset, length).first;
    if (current != free_.begin()) {
      auto before = std::prev(current);
      if (before->first + before->second == current->first) {
        before->second += current->second;
        free_.erase(current);
        current = before;
      }
    }
    auto after = std::next(current);
    if (after != free_.end() &&
        current->first + current->second == after->first) {
      current->second += after->second;
      free_.erase(after);
    }
  }

  size_t peak() const { return peak_bytes_; }

 private:
  uint8_t* base_;
  size_t capacity_, live_bytes_ = 0, peak_bytes_ = 0;
  std::map<size_t, size_t> free_, live_;
  std::mutex mutex_;
};

class NativeANS {
 public:
  NativeANS(uintptr_t stream, uintptr_t workspace, size_t workspace_bytes,
            uintptr_t device_size, uintptr_t host_size,
            const std::vector<size_t>& sizes)
      : stream_(reinterpret_cast<cudaStream_t>(stream)),
        bank_(std::make_shared<FixedBank>(workspace, workspace_bytes)),
        device_size_(reinterpret_cast<size_t*>(device_size)),
        host_size_(reinterpret_cast<size_t*>(host_size)) {
    // These match nvcomp.Codec(algorithm="ANS") defaults in installed 6.0.
    manager_ = std::make_unique<nvcomp::ANSManager>(
        65536, nvcompBatchedANSCompressDefaultOpts,
        nvcompBatchedANSDecompressDefaultOpts, stream_,
        nvcomp::ChecksumPolicy::NoComputeNoVerify,
        nvcomp::ExecutionPolicy::Concurrency,
        nvcomp::BitstreamKind::NVCOMP_NATIVE);
    auto bank = bank_;
    manager_->set_scratch_allocators(
        [bank](size_t bytes) { return bank->allocate(bytes); },
        [bank](void* pointer, size_t bytes) { bank->release(pointer, bytes); });
    add_sizes(sizes);
  }

  void add_sizes(const std::vector<size_t>& sizes) {
    for (size_t size : sizes) {
      if (compression_.count(size)) continue;
      auto config = manager_->configure_compression(size);
      decompression_.emplace(size, manager_->configure_decompression(config));
      compression_.emplace(size, std::move(config));
    }
  }

  void encode(uintptr_t source, uintptr_t output, size_t size) {
    manager_->compress(reinterpret_cast<const uint8_t*>(source),
                       reinterpret_cast<uint8_t*>(output),
                       compression_.at(size), device_size_);
    checked(cudaMemcpyAsync(host_size_, device_size_, sizeof(size_t),
                            cudaMemcpyDeviceToHost, stream_));
  }

  void decode(uintptr_t source, uintptr_t output, size_t size) {
    manager_->decompress(reinterpret_cast<uint8_t*>(output),
                         reinterpret_cast<const uint8_t*>(source),
                         decompression_.at(size));
  }

  size_t compressed_size() const { return *host_size_; }
  size_t max_output(size_t size) const {
    return compression_.at(size).max_compressed_buffer_size;
  }
  size_t workspace_peak() const { return bank_->peak(); }
  void close() {
    if (manager_) {
      checked(cudaStreamSynchronize(stream_));
      manager_->deallocate_gpu_mem();
      manager_.reset();
    }
  }

 private:
  cudaStream_t stream_;
  std::shared_ptr<FixedBank> bank_;
  size_t* device_size_;
  size_t* host_size_;
  std::map<size_t, nvcomp::CompressionConfig> compression_;
  std::map<size_t, nvcomp::DecompressionConfig> decompression_;
  std::unique_ptr<nvcomp::ANSManager> manager_;
};


}  // namespace kvserve_fused_native
