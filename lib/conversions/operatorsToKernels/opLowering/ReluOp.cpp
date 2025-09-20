#include "conversions/OperatorsToKernels/opLowering.h"
#include "dialects/kernels/IR/kernels.h"
#include "mlir/IR/Attributes.h"

using namespace mlir;
using namespace llvm;
namespace tbc::ops {
LogicalResult ReluOpLowering::matchAndRewrite(tbc::ops::ReluOp op,
                                              PatternRewriter &rewriter) const {
  auto input = op.getInput();
  std::vector<NamedAttribute> attrs;
  auto outputType = op.getOutput().getType();
  attrs.push_back(
      rewriter.getNamedAttr("mode", rewriter.getStringAttr("RELU")));
  rewriter.replaceOpWithNewOp<tbc::kls::ActiveOp>(op, outputType, input, attrs);
  return success();
}
} // namespace tbc::ops
