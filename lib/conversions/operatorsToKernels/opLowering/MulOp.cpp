#include "conversions/OperatorsToKernels/opLowering.h"

using namespace mlir;
using namespace llvm;
namespace tbc::ops{
  LogicalResult MulOpLowering::matchAndRewrite(tbc::ops::MulOp op,
                                PatternRewriter &rewriter) const {
  auto input = op->getOperands();
  auto outputType = op.getOutput().getType();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("mode", rewriter.getStringAttr("Mul")));
  rewriter.replaceOpWithNewOp<tbc::kls::EltWiseOp>(op, outputType, input,
                                                        attrs);
  return success();
  }
}
