include(AddMLIRPython)

include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

set(TBC_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tbc")
set(TBC_PYTHON_PACKAGES_DIR "${CMAKE_BINARY_DIR}/python")
set(TBC_PYTHON_EXTRA_MODULES)
set(TBC_PYTHON_EXTRA_LIBS)
################################################################################
# Structural groupings.
################################################################################
declare_mlir_python_sources(TBCPythonSources)
declare_mlir_python_sources(TBCPythonSources.Dialects
  ROOT_DIR "${TBC_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT TBCPythonSources
)
################################################################################
# Dialect bindings
################################################################################
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TBCPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tbc"
  TD_FILE dialects/operators.td
  SOURCES dialects/operators.py
  DIALECT_NAME operators)


declare_mlir_python_extension(TBCPythonExtension.RegisterEverything
  MODULE_NAME _mlirRegisterEverything
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tbc"
  SOURCES
    registAll.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIConversion
    MLIRCAPITransforms
    TBCCAPI
)
###################################################
#extra bindings
###################################################
if(ENABLE_TBC_PYTHON_EXTRA)

  list(APPEND TBC_PYTHON_EXTRA_MODULES
      MLIRPythonSources.Dialects.pdl
      MLIRPythonSources.Dialects.transform
      MLIRPythonSources.Dialects.transform.extras
      MLIRPythonSources.Dialects.transform.interpreter

  )

  list(APPEND TBC_PYTHON_EXTRA_LIBS
      MLIRCAPIPDL
      MLIRCAPITransformDialect)
endif()

################################################################################
# Common CAPI dependency DSO.
# All python extensions must link through one DSO which exports the CAPI, and
# this must have a globally unique name amongst all embeddors of the python
# library since it will effectively have global scope.
################################################################################

add_mlir_python_common_capi_library(TBCPythonCAPI
  INSTALL_COMPONENT TBCPythonModules
  INSTALL_DESTINATION python/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    TBCPythonSources
    TBCPythonExtension.RegisterEverything
    MLIRPythonSources.Core
    ${TBC_PYTHON_EXTRA_MODULES}
  EMBED_LIBS
   ${TBC_PYTHON_EXTRA_LIBS}
)

################################################################################
# The fully assembled package of modules.
################################################################################

add_mlir_python_modules(TBCPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python/mlir"
  INSTALL_PREFIX "python/mlir/"
  DECLARED_SOURCES
    TBCPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.func
    TBCPythonExtension.RegisterEverything
  COMMON_CAPI_LINK_LIBS
    TBCPythonCAPI
)

install(
  DIRECTORY ./ DESTINATION python FILES_MATCHING
  USE_SOURCE_PERMISSIONS
  PATTERN "*.py"
  PATTERN "*.js"
  PATTERN "*.html"
  PATTERN "xlsx_to_py.py" EXCLUDE
  PATTERN "test_*" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".pytest_cache" EXCLUDE
  PATTERN "tbc" EXCLUDE
)


install(CODE "
  execute_process(COMMAND find \"\${CMAKE_INSTALL_PREFIX}/python\" -type d -empty -delete)
")
