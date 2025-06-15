#include"capi/registAll.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "dialects/operators/IR/operator.h"


void mlirRegisterAllDialects(MlirDialectRegistry registry) {
  static_cast<mlir::DialectRegistry *>(registry.ptr)
      ->insert<mlir::func::FuncDialect, tbc::ops::OperatorDialect>();
}
void register_llvm_translations(MlirContext &context) {

}

