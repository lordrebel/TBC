#ifndef TBC_KERNEL_DIALECT_TARGET_PASS
#define TBC_KERNEL_DIALECT_TARGET_PASS

include "mlir/Pass/PassBase.td"

def ExtraOptimizeNpuV1Pass : Pass<"extra-optimize-for-npu-v1", "ModuleOp"> {
  let summary = "extra optimize for npu v1 target";
  let constructor = "createExtraOptimizeNpuV1Pass()";
  let dependentDialects = ["KernelDialect"];
  let options = [];
}

#endif 
