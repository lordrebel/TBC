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

def TargetDependentPass : Pass<"target-depend-pass", "ModuleOp"> {
  let summary = "target dependent pass for module op";
  let constructor = "createTargetDependentPass()";
  let dependentDialects = ["KernelDialect"];
  let options = [];
}

def OpReOrderPass : Pass<"op-reorder", "mlir::func::FuncOp"> {
  let summary = "reorder ops for better performance";
  let constructor = "createOpreorderPass()";
  let dependentDialects = ["KernelDialect"];
  let options = [
    Option<"algo", "algo", "std::string",/*default=*/[{std::string("RPO")}],
           "the sort order of reorder pass">,

  ];
}

#endif
