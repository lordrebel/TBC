#include "capi/registAll.h"

#include "dialects/operators/IR/operator.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/MLIRContext.h"


void mlirRegisterAllDialects(MlirDialectRegistry registry) {
  static_cast<mlir::DialectRegistry *>(registry.ptr)
      ->insert<mlir::func::FuncDialect, tbc::ops::OperatorDialect,
               mlir::pdl::PDLDialect,mlir::pdl_interp::PDLInterpDialect ,mlir::transform::TransformDialect>();
}
void register_llvm_translations(MlirContext &context) {}
