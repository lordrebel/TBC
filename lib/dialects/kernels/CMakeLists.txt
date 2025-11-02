file(GLOB _sources
          IR/*.cpp
          interfaces/*.cpp
          transforms/*.cpp
          transforms/target/*.cpp
    )

add_mlir_library(TBCKernels
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/dialects/kernels
	${PROJECT_SOURCE_DIR}/include/dialects/operators

  DEPENDS
  MLIRKernelAttrGen
	MLIROperatorsIncGen
  MLIRKernelsIncGen
  MLIRKernelDialectPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRParser
  MLIRFuncDialect
  MLIRRewrite
  MLIRTransforms
	MLIRTBCSupport

)
