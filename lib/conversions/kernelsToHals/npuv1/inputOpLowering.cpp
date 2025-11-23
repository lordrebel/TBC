#include "conversions/KernelsToHals/npuV1Lowering.h"
#include "dialects/hals/IR/hals.h"
#include "dialects/hals/IR/halsStructs.h"

namespace tbc::npuv1 {
LogicalResult
InputOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  auto inputOp = cast<tbc::ops::InputOp>(op);
  auto loc = inputOp.getLoc();
  if (isa<RankedTensorType>(operands[0].getType())) {
    llvm::errs() << "InputOpLowering returnType:" << operands[0].getType()
                 << "\n";
    llvm_unreachable("invalid");

  } else {
    auto ori_type =inputOp.getOutput().getType();
    auto returnType = typeConverter->convertType(ori_type);
    auto newOp=rewriter.create<tbc::hals::InputOp>(loc, returnType,operands, inputOp->getAttrs());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
  
  return failure();
};
} // namespace tbc::npuv1
