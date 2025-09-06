#ifndef  TBC_COVERSION_PASS
#define  TBC_COVERSION_PASS
include "mlir/Pass/PassBase.td"
//===----------------------------------------------------------------------===//
// OperatorsToKernels
//===----------------------------------------------------------------------===//
def ConvertOperatorsToKernels : Pass<"convert-operators-to-kernels", "ModuleOp"> {
  let summary = "Convert operators Ops to kernels Ops";
  let constructor = "tbc::createConvertOperatorsToKernels()";
  let dependentDialects = ["tbc::ops::OperatorDialect", "tbc::kls::KernelDialect"];
  let options = [
  ];
}

//===----------------------------------------------------------------------===//
// KernelsToHals
//===----------------------------------------------------------------------===//
def ConvertKernelsToHals : Pass<"convert-kernels-to-hals", "ModuleOp"> {
  let summary = "Convert kernels Ops to hal Ops";
  let constructor = "tbc::createConvertKernelsToHals()";
  let dependentDialects = ["tbc::hals::HalDialect", "tbc::kls::KernelDialect"];
  let options = [
  ];
}


#endif // TBC_COVERSION_PASS
