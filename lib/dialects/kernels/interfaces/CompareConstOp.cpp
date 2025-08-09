#include "dialects/kernels/IR/kernels.h"
#include "support/module.h"

using namespace mlir;
namespace tbc::kls { 
  llvm::LogicalResult CompareConstOp::verify() {
    if(getMode() =="Not"){
      LOGE<<"compareConstop for mode:Not not support\n";
      return failure();
    }
    return success();

  }

}
