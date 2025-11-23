#include "conversions/KernelsToHals/npuV1Lowering.h"
#include "dialects/hals/IR/hals.h"
#include "dialects/hals/IR/halsStructs.h"

namespace tbc::npuv1 {
LogicalResult
WeightOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
  auto weightOp = cast<tbc::ops::WeightOp>(op);
  auto loc = weightOp.getLoc();
  auto ori_type =
      cast_or_null<RankedTensorType>(weightOp.getOutput().getType());
  if (!ori_type) {
    llvm::errs() << "WeightOpLowering returnType:" << weightOp.getType()
                 << "\n";
    llvm_unreachable("invalid");
  }
  auto returnType = typeConverter->convertType(ori_type);
  auto newOp = rewriter.create<tbc::hals::WeightOp>(loc, returnType, operands,
                                                    weightOp->getAttrs());
  rewriter.replaceOp(op, newOp.getResult());
  return success();
}
} // namespace tbc::npuv1
