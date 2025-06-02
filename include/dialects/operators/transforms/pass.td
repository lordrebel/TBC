#ifndef TBC_DIALECT_PASS
#define TBC_DIALECT_PASS

include "mlir/Pass/PassBase.td"
def Init : Pass<"init", "ModuleOp"> {
  let summary = "init module";
  let constructor = "createInitPass()";
  let dependentDialects = ["OperatorDialect"];
  let options = [
    Option<"freq", "freq", "uint64_t", /*default=*/"",
           "chip frequance">,
  ];
}

def Deinit : Pass<"deinit", "ModuleOp"> {
  let summary = "deinit module";
  let constructor = "createDeinitPass()";
  let dependentDialects = ["OperatorDialect"];
}

def ShapeInfer : Pass<"shape-infer", "ModuleOp"> {
  let summary = "do shape inference for each op";
  let constructor = "createShapeInferPass()";
  let dependentDialects = ["OperatorDialect"];
}
#endif
