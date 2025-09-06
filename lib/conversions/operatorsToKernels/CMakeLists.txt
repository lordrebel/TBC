file(GLOB _sources
          *.cpp
    )

add_mlir_conversion_library(TbcOperatorsToKernels
      ${_sources}

      ADDITIONAL_HEADER_DIRS
      ${PROJECT_SOURCE_DIR}/include/dialects/operators
			${PROJECT_SOURCE_DIR}/include/dialects/kernels

      DEPENDS
      TBCMlirConversionPassIncGen

      LINK_LIBS PUBLIC
			TBCKernels
			TBCOperators
      
      )
