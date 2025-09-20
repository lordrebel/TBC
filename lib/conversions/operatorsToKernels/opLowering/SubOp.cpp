#include "conversions/OperatorsToKernels/opLowering.h"

using namespace mlir;
using namespace llvm;
namespace tbc::ops {
LogicalResult SubOpLowering::matchAndRewrite(tbc::ops::SubOp op,
                                             PatternRewriter &rewriter) const {
  auto input = op->getOperands();
  auto outputType = op.getOutput().getType();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("mode", rewriter.getStringAttr("Sub")));
  rewriter.replaceOpWithNewOp<tbc::kls::EltWiseOp>(op, outputType, input,
                                                   attrs);
  return success();
}
} // namespace tbc::ops
