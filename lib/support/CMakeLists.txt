file(GLOB _sources *.cpp )

add_llvm_library(MLIRTBCSupport
  ${_sources}

  DEPENDS
  UtilsAttrGen
  MLIROperatorsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  cnpy
)
