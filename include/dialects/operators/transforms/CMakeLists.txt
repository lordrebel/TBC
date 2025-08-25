set(LLVM_TARGET_DEFINITIONS pass.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name op)

if(BUILD_DOC)
mlir_tablegen(pass.md -gen-pass-doc)

install(FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/pass.md
    DESTINATION doc/IR/operator
    COMPONENT documentation
    OPTIONAL  # 如果文件不存在也不报错
)
endif()

add_public_tablegen_target(MLIROperatorDialectPassIncGen)


add_subdirectory(platform)
