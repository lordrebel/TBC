#pragma once
#include "mlir/IR/OpDefinition.h"

namespace tbc {

// and outputs type is the same with the first input type
void common_type_inference(mlir::Operation *op);

// binary operation with the implicit broadcast
void broadcast_type_inference(mlir::Operation *op);

} // namespace tbc

#include "interfaces/TypeInferInterface.h.inc"
