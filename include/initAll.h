#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"          
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
namespace tbc {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();

} // namespace tpu_mlir
