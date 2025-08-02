add_mlir_dialect(Kernels kernels)
#add_mlir_doc(TpuOps TpuOps ./ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS kernels.td)
mlir_tablegen(KernelEnum.h.inc -gen-enum-decls)
mlir_tablegen(KernelEnum.cpp.inc -gen-enum-defs)
mlir_tablegen(KernelAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(KernelAttr.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRKernelAttrGen)
