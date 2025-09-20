#include "conversions/OperatorsToKernels/opLowering.h"
#include "dialects/operators/IR/operator.h"

using namespace mlir;
using namespace llvm;
namespace tbc::ops {
LogicalResult
ConcatOpLowering::matchAndRewrite(tbc::ops::ConcatOp op,
                                  PatternRewriter &rewriter) const {
  auto input = op->getOperands();
  auto attrs = op->getAttrs();
  auto outputType = op.getOutput().getType();
  rewriter.replaceOpWithNewOp<tbc::kls::EltWiseConstOp>(op, outputType, input,
                                                        attrs);
  return success();
}
} // namespace tbc::ops
