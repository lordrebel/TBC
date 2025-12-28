# 1. 告诉 CMake 到哪里找这些 deps
list(APPEND CMAKE_PREFIX_PATH
  ${ORTOOLS_ROOT}
  ${ORTOOLS_ROOT}/lib/cmake
  ${ORTOOLS_ROOT}/lib/cmake/absl
  ${ORTOOLS_ROOT}/lib/cmake/protobuf
)

# 2. 依赖逐个找
find_package(BZip2 REQUIRED)
find_package(Protobuf REQUIRED)
find_package(absl REQUIRED)
