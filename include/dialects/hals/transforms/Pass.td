#ifndef TBC_HAL_DIALECT_PASS
#define TBC_HAL_DIALECT_PASS

include "mlir/Pass/PassBase.td"

def AssginTensorInfos : Pass<"assign-tensorInfo", "ModuleOp"> {
  let summary = "assign compile phase to module";
  let constructor = "createAssginTensorInfosPass()";
  let dependentDialects = ["HalDialect"];
  let options = [
  ];
}



#endif
