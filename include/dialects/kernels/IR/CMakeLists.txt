add_mlir_dialect(Kernels kernels)

set(LLVM_TARGET_DEFINITIONS Kernels.td)

if(BUILD_DOC)
mlir_tablegen(kernels.md -gen-op-doc)
mlir_tablegen(kernelDialect.md -gen-dialect-doc)
mlir_tablegen(kernelTypes.md -gen-typedef-doc)
mlir_tablegen(kernelAttrs.md -gen-attrdef-doc)

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/kernels.md
    ${CMAKE_CURRENT_BINARY_DIR}/kernelDialect.md
    ${CMAKE_CURRENT_BINARY_DIR}/kernelTypes.md
    ${CMAKE_CURRENT_BINARY_DIR}/kernelAttrs.md
    DESTINATION doc/IR/kernel
    COMPONENT documentation
    OPTIONAL  # 如果文件不存在也不报错
)
endif()

mlir_tablegen(KernelEnum.h.inc -gen-enum-decls)
mlir_tablegen(KernelEnum.cpp.inc -gen-enum-defs)
mlir_tablegen(KernelAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(KernelAttr.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRKernelAttrGen)
