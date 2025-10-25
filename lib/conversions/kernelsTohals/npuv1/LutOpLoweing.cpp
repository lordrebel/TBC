#include "conversions/KernelsToHals/npuV1Lowering.h"
#include "dialects/hals/IR/hals.h"
#include "dialects/hals/IR/halsStructs.h"

namespace tbc::npuv1 {
LogicalResult
LutOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter) const {
  auto lutOp = cast<tbc::kls::LutOp>(op);
  auto loc = lutOp.getLoc();
  auto ori_type = cast_or_null<RankedTensorType>(lutOp.getOutput().getType());
  if (!ori_type) {
    llvm::errs() << "LutOpLowering returnType:" << lutOp.getType() << "\n";
    llvm_unreachable("invalid");
  }
  auto returnType = typeConverter->convertType(ori_type);
  auto newOp = rewriter.create<tbc::hals::LutOp>(loc, returnType, operands,
                                                 lutOp->getAttrs());
  rewriter.replaceOp(op, newOp.getResult());
  return success();
}
}; // namespace tbc::npuv1
