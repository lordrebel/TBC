#include "conversions/OperatorsToKernels/opLowering.h"
#include "dialects/kernels/IR/kernels.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace llvm;
namespace tbc::ops {
LogicalResult ConvOpLowering::matchAndRewrite(tbc::ops::ConvOp op,
                                              PatternRewriter &rewriter) const {
  auto param = op.parseParam();
  if (param.dims != 2) {
    LOGE << "only support 2d conv lowering for now \n";
    return failure();
  }
  auto input = op.getOperands();
  auto outputType = op.getOutput().getType();
  std::vector<NamedAttribute> attrs;
  auto winograd = op.getDoWinogradAttr();
  if (winograd == nullptr) {
    winograd = rewriter.getBoolAttr(false);
  }
  attrs.push_back(
      rewriter.getNamedAttr("kernel_shape", op.getKernelShapeAttr()));
  attrs.push_back(rewriter.getNamedAttr("strides", op.getStridesAttr()));
  attrs.push_back(rewriter.getNamedAttr("pads", op.getPadsAttr()));
  attrs.push_back(rewriter.getNamedAttr("dilations", op.getDilationsAttr()));
  attrs.push_back(rewriter.getNamedAttr("do_winograd", winograd));
  attrs.push_back(rewriter.getNamedAttr("group", op.getGroupAttr()));

  rewriter.replaceOpWithNewOp<tbc::kls::Conv2dOp>(op, outputType, input, attrs);
  return success();
}

} // namespace tbc::ops
