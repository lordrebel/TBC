#include "conversions/KernelsToHals/npuV1Lowering.h"
#include "dialects/hals/IR/hals.h"
#include "dialects/hals/IR/halsStructs.h"
#include "dialects/kernels/IR/kernels.h"
#include "mlir/IR/Location.h"

namespace tbc::npuv1 {
LogicalResult
EltWiseOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter) const {
  auto elitwiseOp = cast<tbc::kls::EltWiseOp>(op);
  auto loc = elitwiseOp.getLoc();
  auto ori_type = cast_or_null<RankedTensorType>(elitwiseOp.getOutput().getType());
  if (!ori_type) {
    llvm::errs() << "elitwiseOpLowering returnType:" << elitwiseOp.getType() << "\n";
    llvm_unreachable("invalid");
  }
  auto returnType = typeConverter->convertType(ori_type);
  auto newOp = rewriter.create<tbc::hals::EltwiseOp>(loc, returnType, operands,
                                                 elitwiseOp->getAttrs());
  rewriter.replaceOp(op, newOp.getResult());
  return success();
}
}; // namespace tbc::npuv1
