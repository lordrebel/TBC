add_subdirectory(cnpy)
if(TARGET cnpy)
    target_compile_options(cnpy PRIVATE -fexceptions)
endif()
