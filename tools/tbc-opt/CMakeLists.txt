
set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

set(LIBS
  MLIRTosaDialect
  MLIRFuncDialect
  MLIROptLib
  TBCInitAll
  )

add_llvm_executable(tbc-opt
  main.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(tbc-opt PRIVATE ${LIBS})
llvm_update_compile_flags(tbc-opt)

mlir_check_all_link_libraries(tbc-opt)

install(TARGETS tbc-opt DESTINATION bin)
