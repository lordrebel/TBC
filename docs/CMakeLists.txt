add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/3rdparty/docgen/tools/pandoc/bin/pandoc
    COMMAND ${CMAKE_SOURCE_DIR}/3rdparty/docgen/tools/install_pandoc.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/3rdparty/docgen/tools
    COMMENT "installing pandoc..."
    VERBATIM
)

add_custom_target(install_pandoc
    DEPENDS ${CMAKE_SOURCE_DIR}/3rdparty/docgen/tools/pandoc/bin/pandoc
)

add_custom_target(generate_docs ALL
    COMMAND ${CMAKE_SOURCE_DIR}/3rdparty/docgen/build.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "generating docs..."
    VERBATIM
)

add_dependencies(generate_docs install_pandoc)

install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/design.docx
    ${CMAKE_CURRENT_SOURCE_DIR}/dev.docx
    DESTINATION doc
    COMPONENT documentation
    OPTIONAL  # 如果文件不存在也不报错
)
