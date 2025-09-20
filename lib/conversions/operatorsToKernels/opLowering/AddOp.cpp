#include "conversions/OperatorsToKernels/opLowering.h"

using namespace mlir;
using namespace llvm;
namespace tbc::ops {
LogicalResult AddOpLowering::matchAndRewrite(tbc::ops::AddOp op,
                                             PatternRewriter &rewriter) const {
  auto input = op->getOperands();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("mode", rewriter.getStringAttr("Add")));
  auto outputType = op.getOutput().getType();
  rewriter.replaceOpWithNewOp<tbc::kls::EltWiseConstOp>(op, outputType, input);
  return success();
}
} // namespace tbc::ops
