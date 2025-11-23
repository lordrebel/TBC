#include "conversions/KernelsToHals/npuV1Lowering.h"
#include "dialects/hals/IR/hals.h"
#include "dialects/hals/IR/halsStructs.h"

namespace tbc::npuv1 {
LogicalResult
ReturnOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  if (isa<RankedTensorType>(operands[0].getType())) {
    llvm::errs() << "ReturnOpLowering returnType:" << operands[0].getType()
                 << "\n";
    llvm_unreachable("invalid");

  } else {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, operands);
    return success();
  }
  
  return failure();
};
} // namespace tbc::npuv1
