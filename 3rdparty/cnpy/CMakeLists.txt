file(GLOB _sources *.cpp)
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

add_llvm_library(cnpy
  ${_sources}
  LINK_LIBS PUBLIC
  ${ZLIB_LIBRARIES}
)
