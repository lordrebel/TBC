set(LLVM_TARGET_DEFINITIONS TargetPass.td)
mlir_tablegen(TargetPass.h.inc -gen-pass-decls -name target)
add_public_tablegen_target(MLIRTargetPassIncGen)
