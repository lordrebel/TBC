set(LLVM_TARGET_DEFINITIONS PlatformPass.td)
mlir_tablegen(PlatformPass.h.inc -gen-pass-decls -name platform)
add_public_tablegen_target(MLIRPlatformPassIncGen)
