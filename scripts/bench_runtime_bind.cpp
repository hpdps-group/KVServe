#include <torch/extension.h>

void bench_fwht(int64_t, int64_t, int64_t, int, int, int, int, int64_t);
void bench_quant(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                 int, int, int, int, int, int, int, int, int64_t);
void bench_dequant(int64_t, int64_t, int64_t, int64_t, int, int, int, int,
                   int, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwht", &bench_fwht, py::call_guard<py::gil_scoped_release>());
  m.def("quant", &bench_quant, py::call_guard<py::gil_scoped_release>());
  m.def("dequant", &bench_dequant, py::call_guard<py::gil_scoped_release>());
}
