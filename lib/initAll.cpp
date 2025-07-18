#include "initAll.h"
#include "dialects/operators/IR/operator.h"
#include "dialects/operators/transforms/pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Transforms/Passes.h"
using namespace mlir;
namespace tbc {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect, mlir::pdl::PDLDialect,
                  mlir::pdl_interp::PDLInterpDialect,
                  mlir::transform::TransformDialect, ops::OperatorDialect>();
}

void registerAllPasses() {
  registerCanonicalizer();
  //  mlir::registerConversionPasses();
  ops::registeropPasses();
}
} // namespace tbc
