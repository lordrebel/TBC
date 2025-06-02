add_mlir_dialect(Operators operators)
#add_mlir_doc(TopOps TopOps ./ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS Operators.td)
mlir_tablegen(operatorAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(operatorAttr.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIROperatorDialectAttrGen)
