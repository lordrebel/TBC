set(LLVM_TARGET_DEFINITIONS Utils.td)
mlir_tablegen(UtilsEnum.h.inc -gen-enum-decls)
mlir_tablegen(UtilsEnum.cpp.inc -gen-enum-defs)
add_public_tablegen_target(UtilsAttrGen)
