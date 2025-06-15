#ifndef TBC_DIALECT_PASS
#define TBC_DIALECT_PASS

include "mlir/Pass/PassBase.td"
def Init : Pass<"init", "ModuleOp"> {
  let summary = "init module";
  let constructor = "createInitPass()";
  let dependentDialects = ["OperatorDialect"];

}

def Deinit : Pass<"deinit", "ModuleOp"> {
  let summary = "deinit module";
  let constructor = "createDeinitPass()";
  let dependentDialects = ["OperatorDialect"];
}

def AssginCompilePhase : Pass<"assign-compile-phase", "ModuleOp"> {
  let summary = "assign compile phase to module";
  let constructor = "createAssignCompilePhasePass()";
  let dependentDialects = ["OperatorDialect"];
  let options = [
    Option<"compile_phase", "compile_phase", "std::string",/*default=*/[{std::string("IMPORTED")}],
           "set compile_phase">,
  ];
}

def ShapeInfer : Pass<"shape-infer", "ModuleOp"> {
  let summary = "do shape inference for each op";
  let constructor = "createShapeInferPass()";
  let dependentDialects = ["OperatorDialect"];
}
#endif
