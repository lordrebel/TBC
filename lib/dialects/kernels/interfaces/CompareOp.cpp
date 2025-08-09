#include "dialects/kernels/IR/kernels.h"
#include "support/module.h"

using namespace mlir;
namespace tbc::kls { 
  llvm::LogicalResult CompareOp::verify() {
    if(getMode() =="Not" &&!tbc::module::isNone(getRight())){
      LOGE<<"compareop for mode:Not only has one operand but get two\n";
      return failure();
    }
    return success();

  }

}
