#include "conversions/OperatorsToKernels/opLowering.h"

using namespace mlir;
using namespace llvm;
namespace tbc::ops {
LogicalResult
MulConstOpLowering::matchAndRewrite(tbc::ops::MulConstOp op,
                                    PatternRewriter &rewriter) const {
  auto input = op.getInput();
  auto const_val = op.getConstVal();
  auto outputType = op.getOutput().getType();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("mode", rewriter.getStringAttr("Mul")));
  attrs.push_back(rewriter.getNamedAttr(
      "const_val", rewriter.getF32FloatAttr(const_val.convertToFloat())));
  rewriter.replaceOpWithNewOp<tbc::kls::EltWiseConstOp>(op, outputType, input,
                                                        attrs);
  return success();
}
} // namespace tbc::ops
