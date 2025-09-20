#include "conversions/OperatorsToKernels/opLowering.h"
#include "dialects/kernels/IR/kernels.h"
#include "dialects/operators/IR/operator.h"

using namespace mlir;
using namespace llvm;
namespace tbc::ops {
LogicalResult ActivationLutOpLowering::matchAndRewrite(tbc::ops::ActivationLutOp op,
                                             PatternRewriter &rewriter) const {
  auto input = op->getOperands();
  std::vector<NamedAttribute> attrs;
  auto bin_mode=op.getBinMode();
  auto sig_mode=op.getSigMode();
  auto cal_mode=op.getCalMode();
  attrs.push_back(rewriter.getNamedAttr("lut_attr", kls::LutAttrAttr::get(rewriter.getContext(), sig_mode, bin_mode, cal_mode)));
  auto outputType = op.getOutput().getType();
  rewriter.replaceOpWithNewOp<tbc::kls::EltWiseConstOp>(op, outputType, input,attrs);
  return success();
}
} // namespace tbc::ops
