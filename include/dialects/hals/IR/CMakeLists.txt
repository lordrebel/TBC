add_mlir_dialect(Hals hals)

set(LLVM_TARGET_DEFINITIONS Hals.td)

if(BUILD_DOC)
mlir_tablegen(hals.md -gen-op-doc)
mlir_tablegen(halDialect.md -gen-dialect-doc)
mlir_tablegen(halTypes.md -gen-typedef-doc)
mlir_tablegen(halAttrs.md -gen-attrdef-doc)

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/hals.md
    ${CMAKE_CURRENT_BINARY_DIR}/halDialect.md
    ${CMAKE_CURRENT_BINARY_DIR}/halTypes.md
    ${CMAKE_CURRENT_BINARY_DIR}/halAttrs.md
    DESTINATION doc/IR/hal
    COMPONENT documentation
    OPTIONAL  # 如果文件不存在也不报错
)
endif()


mlir_tablegen(HalEnum.h.inc -gen-enum-decls)
mlir_tablegen(HalEnum.cpp.inc -gen-enum-defs)
mlir_tablegen(HalAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(HalAttr.cpp.inc -gen-attrdef-defs)
mlir_tablegen(HalTypes.h.inc -gen-typedef-decls)
mlir_tablegen(HalTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRHalAttrGen)
