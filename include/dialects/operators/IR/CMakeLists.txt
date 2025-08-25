add_mlir_dialect(Operators operators)

set(LLVM_TARGET_DEFINITIONS Operators.td)

if(BUILD_DOC)
mlir_tablegen(operators.md -gen-op-doc)
mlir_tablegen(operatorDialect.md -gen-dialect-doc)
mlir_tablegen(operatorTypes.md -gen-typedef-doc)
mlir_tablegen(operatorAttrs.md -gen-attrdef-doc)

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/operators.md
    ${CMAKE_CURRENT_BINARY_DIR}/operatorDialect.md
    ${CMAKE_CURRENT_BINARY_DIR}/operatorTypes.md
    ${CMAKE_CURRENT_BINARY_DIR}/operatorAttrs.md
    DESTINATION doc/IR/operator
    COMPONENT documentation
    OPTIONAL  # 如果文件不存在也不报错
)
endif()
mlir_tablegen(operatorAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(operatorAttr.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIROperatorDialectAttrGen)


