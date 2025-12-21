file(GLOB _sources
          IR/*.cpp
          interfaces/*.cpp
          transforms/*.cpp
    )

add_mlir_library(TBCHals
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/dialects/kernels
	${PROJECT_SOURCE_DIR}/include/dialects/hals

  DEPENDS
	MLIRHalsIncGen
	MLIRHalAttrGen
  MLIRHalDialectPassIncGen
  MLIRMemInfoInferInterfaceIncGen
  MLIRPackWeightInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRParser
  MLIRFuncDialect
  MLIRRewrite
  MLIRTransforms
	MLIRTBCSupport
  MLIRMemInfoInferInterface
  MLIRPackWeightInterface

)
