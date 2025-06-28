set(LLVM_TARGET_DEFINITIONS pass.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name op)
add_public_tablegen_target(MLIROperatorDialectPassIncGen)
add_subdirectory(platform)
