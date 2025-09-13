file(GLOB _sources
          *.cpp
          npuv1/*.cpp
          npuv2/*.cpp
    )

add_mlir_conversion_library(TbcKernelsToHals
      ${_sources}

      ADDITIONAL_HEADER_DIRS
      ${PROJECT_SOURCE_DIR}/include/dialects/hals
			${PROJECT_SOURCE_DIR}/include/dialects/kernels

      DEPENDS
      TBCMlirConversionPassIncGen

      LINK_LIBS PUBLIC
			TBCKernels
			TBCHals
      
      )
