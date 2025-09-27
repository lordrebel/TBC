#include "conversions/OperatorsToKernels/opLowering.h"
#include "dialects/kernels/IR/kernels.h"

using namespace mlir;
using namespace llvm;
namespace tbc::ops {
LogicalResult DivOpLowering::matchAndRewrite(tbc::ops::DivOp op,
                                             PatternRewriter &rewriter) const {
  auto input = op.getOperands();
  auto outputType = op.getOutput().getType();
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("mode", rewriter.getStringAttr("Div")));
  rewriter.replaceOpWithNewOp<tbc::kls::EltWiseOp>(op, outputType, input,
                                                        attrs);
  return success();
}
} // namespace tbc::ops
