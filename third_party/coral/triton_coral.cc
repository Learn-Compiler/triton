///-----------------------------------------------------------------------------
// C++插件实现
///-----------------------------------------------------------------------------
#include <nanobind/nanobind.h>

namespace py = nanobind;

void init_triton_coral(py::module_ &m) {
  m.doc() = "Python bindings to the Coral Triton backend";

  // 示例导出接口（后续在此注册 Coral 特有的 Dialect、Passes、LLVM Target 设置等）
  m.def("hello", []() {
    return "Hello from Coral backend!";
  });
}
