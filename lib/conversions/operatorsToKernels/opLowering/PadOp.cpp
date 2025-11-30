#include "conversions/OperatorsToKernels/opLowering.h"
#include "dialects/kernels/IR/kernels.h"

using namespace mlir;
using namespace llvm;
namespace tbc::ops {
/*
    AnyTensor:$input,
  I64ArrayAttr:$paddings,
  DefaultValuedAttr<F64Attr, "0.0">:$val,
  PaddingModeAttr:$mode
*/

/*
    AnyTensor:$input,
  I64ArrayAttr:$paddings,
  DefaultValuedAttr<F64Attr, "0.0">:$val,
  DefaultValuedAttr<Kernel_PaddingModeAttr,
  "tbc::kls::PaddingMode::constant">:$mode, DefaultValuedAttr<BoolAttr,
  "false">:$do_relu, DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit

*/
LogicalResult PadOpLowering::matchAndRewrite(tbc::ops::PadOp op,
                                             PatternRewriter &rewriter) const {
  auto input = op->getOperands();
  auto outputType = op.getOutput().getType();
  auto mode = kls::symbolizePaddingMode(op.getMode());
  if (!mode.has_value()) {
    LOGE << "not support pad mode lowering:" << op.getMode() << "\n";
    return failure();
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("pads", op.getPaddings()));
  attrs.push_back(rewriter.getNamedAttr("val", op.getValAttr()));
  attrs.push_back(rewriter.getNamedAttr(
      "mode", kls::PaddingModeAttr::get(getContext(), mode.value())));
  rewriter.replaceOpWithNewOp<tbc::kls::PadOp>(op, outputType, input, attrs);
  return success();
}

} // namespace tbc::ops
