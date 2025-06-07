#include "mlir/CAPI/Registration.h"
#include "dialects/operators/IR/operator.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Operators, ops, tbc::ops::OperatorDialect)
