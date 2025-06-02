file(GLOB _sources
          IR/*.cpp
          interfaces/*.cpp
          canonicalize/*.cpp
          transforms/*.cpp
    )

add_mlir_library(TBCOperators
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/dialects/operators

  DEPENDS
  MLIROperatorDialectAttrGen
  MLIROperatorsIncGen
  MLIROperatorDialectPassIncGen
  MLIRShapeInferInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRParser
  MLIRFuncDialect
  MLIRRewrite
  MLIRTransforms
	MLIRTBCSupport
  MLIRShapeInferInterface
)
