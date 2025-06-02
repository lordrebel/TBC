#pragma once

#include "mlir/IR/Dialect.h"

namespace tbc {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();

} // namespace tpu_mlir
