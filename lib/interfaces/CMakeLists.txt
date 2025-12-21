set(
  LLVM_OPTIONAL_SOURCES
  ShapeInferInterface.cpp
  TypeInferInterface.cpp
  MemInfoInferInterface.cpp
  PackWeightInterface.cpp
)

function(add_mlir_interface_library name)
  add_mlir_library(
    MLIR${name}
    ${name}.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

    DEPENDS
    MLIR${name}IncGen

    LINK_LIBS PUBLIC
    MLIRIR
  )
endfunction(add_mlir_interface_library)


add_mlir_interface_library(ShapeInferInterface)
add_mlir_interface_library(TypeInferInterface)
add_mlir_interface_library(MemInfoInferInterface)
add_mlir_interface_library(PackWeightInterface)

