#ifndef TBC_KERNEL_DIALECT_PASS
#define TBC_KERNEL_DIALECT_PASS

include "mlir/Pass/PassBase.td"

def AssginTarget : Pass<"assign-target", "ModuleOp"> {
  let summary = "assign compile phase to module";
  let constructor = "createAssginTargetPass()";
  let dependentDialects = ["KernelDialect"];
  let options = [
    Option<"target", "target", "std::string",/*default=*/[{std::string("ALL")}],
           "set target">,
  ];
}

#endif
