add_mlir_public_c_api_library(TBCCAPI
  dialects.cpp
  registAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  TBCOperators
  TBCKernels
  TBCHals
  MLIRFuncDialect
)
