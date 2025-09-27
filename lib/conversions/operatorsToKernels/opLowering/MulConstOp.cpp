#include "conversions/OperatorsToKernels/opLowering.h"

using namespace mlir;
using namespace llvm;
namespace tbc::ops {
LogicalResult
MulConstOpLowering::matchAndRewrite(tbc::ops::MulConstOp op,
                                    PatternRewriter &rewriter) const {
  auto input = op.getInput();
  auto constant=op.getConstVal().convertToDouble();
  auto outputType = op.getOutput().getType();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("mode", rewriter.getStringAttr("Mul")));
  attrs.push_back(rewriter.getNamedAttr(
      "const_val", rewriter.getF64FloatAttr(constant)));
  rewriter.replaceOpWithNewOp<tbc::kls::EltWiseConstOp>(op, outputType, input,
                                                        attrs);
  return success();
}
} // namespace tbc::ops
