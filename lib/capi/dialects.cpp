#include "mlir/CAPI/Registration.h"
#include "dialects/operators/IR/operator.h"
#include "dialects/kernels/IR/kernels.h"
#include "dialects/hals/IR/hals.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Operators, ops, tbc::ops::OperatorDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Kernels, kls, tbc::kls::KernelDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Hals, hals, tbc::hals::HalDialect)
