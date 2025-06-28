#ifndef TBC_PLATFORM_PASS
#define TBC_PLATFORM_PASS
include "mlir/Pass/PassBase.td"
def PrintOpname : Pass<"PrintOpnamePass", "ModuleOp"> {
  let summary = "just for test,print each ops name";
  let constructor = "createPrintOpNamePass()";
  let dependentDialects = ["OperatorDialect"];
}
#endif
