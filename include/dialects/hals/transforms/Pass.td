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

def PackWeightsPass : Pass<"pack-weights", "FuncOp"> {
  let summary = "pack weights for hals dialect";
  let constructor = "createPackWeightsPass()";
  let dependentDialects = ["HalDialect"];
  let options = [
  ];
}

def FusePackedWeightGroupToOnePass : Pass<"fuse-packed-weight-group-to-one", "FuncOp"> {
  let summary = "fuse packed weight group to one";
  let constructor = "createFusePackedWeightGroupToOnePass()";
  let dependentDialects = ["HalDialect"]; 
}



#endif
