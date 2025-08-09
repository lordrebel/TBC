#include "dialects/kernels/IR/kernels.h"

using namespace mlir;
namespace tbc::kls { 
  llvm::LogicalResult ReduceOp::verify() {
    if((getMode() =="ReduceSum"|| getMode()=="ReduceProd") &&getOutType()=="IDX"){
      LOGE<<"reduceop output index NOT support when mode is ReduceSum or ReduceProd\n";
      return failure();
    }
    return success();

  }

}
