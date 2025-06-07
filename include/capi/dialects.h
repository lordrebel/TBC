#ifndef TPU_MLIR_C_DIALECTS_H
#define TPU_MLIR_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Operators, ops);

#ifdef __cplusplus
}
#endif

#endif // TPU_MLIR_C_DIALECTS_H
