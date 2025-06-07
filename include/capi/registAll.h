#ifndef TPUMLIR_C_REGISTER_EVERYTHING_H
#define TPUMLIR_C_REGISTER_EVERYTHING_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Appends all upstream dialects and extensions to the dialect registry.
MLIR_CAPI_EXPORTED void mlirRegisterAllDialects(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // TPUMLIR_C_REGISTER_EVERYTHING_H
