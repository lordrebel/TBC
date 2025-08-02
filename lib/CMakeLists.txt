add_subdirectory(interfaces)
add_subdirectory(traits)
add_subdirectory(dialects)
add_subdirectory(support)
add_subdirectory(conversions)
add_subdirectory(capi)
#todo(jiahao) finish init all

add_mlir_library(TBCInitAll
  initAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect

  TBCOperators
  TBCKernels
  TBCHals
)

