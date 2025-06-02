#include "dialects/operators/IR/operator.h"
#include "dialects/operators/transforms/pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"
#include"initAll.h"
using namespace mlir;
namespace tbc {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry
      .insert< mlir::func::FuncDialect, ops::OperatorDialect>();
}

void registerAllPasses() {
  registerCanonicalizer();
//  mlir::registerConversionPasses();
  ops::registeropPasses();
}
} // namespace tbc
