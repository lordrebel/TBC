#include "capi/registAll.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"


PYBIND11_MODULE(_mlirRegisterEverything, m) {
  m.doc() = "TBC Dialects Registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);
  });
  m.def("register_llvm_translations", &register_llvm_translations);
}
