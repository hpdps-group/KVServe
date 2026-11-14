// SPDX-License-Identifier: Apache-2.0
#include "native_ans.hpp"
#include "kvs_fused.h"
#include <tuple>
using kvserve_fused_native::NativeANS;
using kvserve_fused_native::checked;

class DeviceGuard {
 public:
  explicit DeviceGuard(int device) { checked(cudaGetDevice(&previous_)); checked(cudaSetDevice(device)); }
  ~DeviceGuard() { cudaSetDevice(previous_); }
 private:
  int previous_;
};

class GraphBudget {
 public:
  GraphBudget(int device, size_t bytes) : device_(device) {
    DeviceGuard guard(device_);
    checked(cudaMalloc(&pointer_, bytes));
  }
  void close() {
    if (pointer_) { DeviceGuard guard(device_); checked(cudaFree(pointer_)); pointer_ = nullptr; }
  }
  ~GraphBudget() { if (pointer_) { try { close(); } catch (...) {} } }
 private:
  int device_;
  void* pointer_ = nullptr;
};

class ExecutionPlan {
 public:
  ExecutionPlan(int64_t cpu, int64_t gpu, int64_t host_ids, int64_t gpu_ids,
                int capacity, int heads, int part, int device, int64_t stream)
      : cpu_(cpu), gpu_(gpu), host_ids_(host_ids), gpu_ids_(gpu_ids),
        capacity_(capacity), heads_(heads), part_(part), device_(device), stream_(stream) {
    if (!cpu || !gpu || !host_ids || !gpu_ids || !stream || capacity < 1 ||
        capacity > 1024 || capacity % 16 || part < 0 || part > 1)
      throw std::invalid_argument("Invalid fused execution plan");
    layers_ = reinterpret_cast<const int64_t*>(cpu_)[0];
    if (layers_ < 1) throw std::invalid_argument("Empty fused execution plan");
  }
  void build(int tokens, bool encode) {
    DeviceGuard guard(device_);
    validate(tokens, expected(tokens));
    const auto key = std::make_pair(encode, tokens);
    if (graphs_.count(key)) throw std::invalid_argument("Duplicate fused graph");
    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    checked(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
      launch_eager(tokens, encode);
      checked(cudaStreamEndCapture(stream, &graph));
      checked(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
      checked(cudaGraphDestroy(graph)); graph = nullptr;
      graphs_.emplace(key, executable);
    } catch (...) {
      cudaStreamCaptureStatus status;
      if (cudaStreamIsCapturing(stream, &status) == cudaSuccess && status != cudaStreamCaptureStatusNone) {
        cudaGraph_t discarded = nullptr;
        cudaStreamEndCapture(stream, &discarded);
        if (discarded) cudaGraphDestroy(discarded);
      }
      if (graph) cudaGraphDestroy(graph);
      if (executable) cudaGraphExecDestroy(executable);
      throw;
    }
  }
  void encode(NativeANS& ans, uintptr_t quant, uintptr_t output, size_t bytes,
              uintptr_t completion, int tokens, uintptr_t aux_source,
              uintptr_t aux_output, size_t aux_bytes) {
    DeviceGuard guard(device_);
    validate(tokens, bytes);
    launch(tokens, true);
    ans.encode(quant, output, bytes);
    if (aux_bytes) {
      if (!aux_source || !aux_output) throw std::invalid_argument("Missing leased auxiliary output");
      checked(cudaMemcpyAsync(reinterpret_cast<void*>(aux_output),
          reinterpret_cast<void*>(aux_source), aux_bytes, cudaMemcpyDeviceToDevice,
          reinterpret_cast<cudaStream_t>(stream_)));
    }
    checked(cudaEventRecord(reinterpret_cast<cudaEvent_t>(completion), reinterpret_cast<cudaStream_t>(stream_)));
  }
  void decode(NativeANS& ans, uintptr_t source, uintptr_t quant, size_t bytes,
              uintptr_t completion, int tokens, bool standard_layout) {
    DeviceGuard guard(device_);
    if (tokens < 1 || tokens > capacity_ ||
        bytes != expected(standard_layout ? capacity_ : tokens))
      throw std::invalid_argument("Invalid fused decode layout or length");
    ans.decode(source, quant, bytes);
    if (standard_layout && tokens != capacity_)
      kvs_fused_decode(cpu_, gpu_, host_ids_, gpu_ids_, tokens, part_, capacity_,
                       heads_, stream_, true);
    else
      launch(tokens, false);
    checked(cudaEventRecord(reinterpret_cast<cudaEvent_t>(completion), reinterpret_cast<cudaStream_t>(stream_)));
  }
  ~ExecutionPlan() { if (!graphs_.empty()) { try { close(); } catch (...) {} } }
  void close() {
    DeviceGuard guard(device_);
    checked(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)));
    for (auto entry : graphs_) checked(cudaGraphExecDestroy(entry.second));
    graphs_.clear();
  }
 private:
  size_t expected(int tokens) const { return layers_ * ((tokens + 15) / 16 * 16) * heads_ * 128; }
  void validate(int tokens, size_t bytes) const {
    if (tokens < 1 || tokens > capacity_ || bytes != expected(tokens))
      throw std::invalid_argument("Fused frame length does not match geometry");
  }
  void launch_eager(int tokens, bool encode) {
    if (encode) kvs_fused_encode(cpu_, gpu_, host_ids_, gpu_ids_, tokens, part_, capacity_, heads_, stream_);
    else kvs_fused_decode(cpu_, gpu_, host_ids_, gpu_ids_, tokens, part_, capacity_, heads_, stream_, false);
  }
  void launch(int tokens, bool encode) {
    auto found = graphs_.find(std::make_pair(encode, tokens));
    if (found == graphs_.end()) launch_eager(tokens, encode);
    else checked(cudaGraphLaunch(found->second, reinterpret_cast<cudaStream_t>(stream_)));
  }
  int64_t cpu_, gpu_, host_ids_, gpu_ids_, stream_, layers_;
  int capacity_, heads_, part_, device_;
  std::map<std::pair<bool,int>, cudaGraphExec_t> graphs_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<NativeANS>(module, "NativeANS")
    .def(py::init<uintptr_t,uintptr_t,size_t,uintptr_t,uintptr_t,const std::vector<size_t>&>(), py::call_guard<py::gil_scoped_release>())
    .def("compressed_size", &NativeANS::compressed_size)
    .def("max_output", &NativeANS::max_output)
    .def("workspace_peak", &NativeANS::workspace_peak)
    .def("close", &NativeANS::close, py::call_guard<py::gil_scoped_release>());
  py::class_<GraphBudget>(module, "GraphBudget")
    .def(py::init<int,size_t>(), py::call_guard<py::gil_scoped_release>())
    .def("close", &GraphBudget::close, py::call_guard<py::gil_scoped_release>());
  py::class_<ExecutionPlan>(module, "ExecutionPlan")
    .def(py::init<int64_t,int64_t,int64_t,int64_t,int,int,int,int,int64_t>())
    .def("build", &ExecutionPlan::build, py::call_guard<py::gil_scoped_release>())
    .def("encode", &ExecutionPlan::encode, py::call_guard<py::gil_scoped_release>())
    .def("decode", &ExecutionPlan::decode, py::call_guard<py::gil_scoped_release>())
    .def("close", &ExecutionPlan::close, py::call_guard<py::gil_scoped_release>());
  module.def("bind_shared", &kvs_fused_bind_shared, py::call_guard<py::gil_scoped_release>());
}
