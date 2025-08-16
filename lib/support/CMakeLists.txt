file(GLOB _sources *.cpp )
file(GLOB _target_sources *.cpp )

add_llvm_library(MLIRTBCSupport
  ${_sources}
  ${_target_sources}

  DEPENDS
  UtilsAttrGen
  MLIROperatorsIncGen
  MLIRKernelsIncGen
  MLIRHalsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  cnpy
)
