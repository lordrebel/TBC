include_directories(${ORTOOLS_INCLUDE_DIRS})
add_executable(ortools-demo ortools_demo.cpp)

set_target_properties(ortools-demo PROPERTIES
    BUILD_RPATH "${ORTOOLS_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN"  # $ORIGIN表示可执行文件所在目录
    INSTALL_RPATH_USE_LINK_PATH FALSE
)

target_link_libraries(ortools-demo PRIVATE ortools::ortools)

install(TARGETS ortools-demo DESTINATION bin/experimental)
install(DIRECTORY ${ORTOOLS_ROOT}/lib/
    DESTINATION bin/experimental
    FILES_MATCHING 
    PATTERN "*.so*"
    PATTERN "cmake" EXCLUDE
    PATTERN "pkgconfig" EXCLUDE
)
