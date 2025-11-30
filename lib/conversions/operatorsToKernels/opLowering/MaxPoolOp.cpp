#include "conversions/OperatorsToKernels/opLowering.h"
#include "dialects/kernels/IR/kernels.h"
#include "dialects/operators/IR/operator.h"
#include "support/module.h"

using namespace mlir;
using namespace llvm;
namespace tbc::ops {

/*

    AnyTensor:$input,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    Kernel_PoolModeAttr:$pool_mode,
    DefaultValuedAttr<Kernel_PaddingModeAttr,"tbc::kls::PaddingMode::no_pad">:$pad_mode,
    DefaultValuedAttr<I64Attr, "0">:$pad_value,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
*/
LogicalResult MaxPoolOpLowering::matchAndRewrite(tbc::ops::MaxPoolOp op,
                                             PatternRewriter &rewriter) const {
  auto input = op->getOperands();
  auto outputType = op.getOutput().getType();
  auto pad=module::getI64Array(op.getPads());
  kls::PaddingMode pad_mode = kls::PaddingMode::no_pad;
  if(count(pad->begin(),pad->end(), 0) != pad->size()){
    pad_mode = kls::PaddingMode::constant;
  }
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("kernel_shape", op.getKernelShapeAttr()));
  attrs.push_back(
      rewriter.getNamedAttr("strides", op.getStridesAttr()));
  attrs.push_back(
      rewriter.getNamedAttr("pads", op.getPadsAttr()));
  attrs.push_back(
      rewriter.getNamedAttr("pool_mode", kls::PoolModeAttr::get(getContext(),
                                                               kls::PoolMode::Max)));
  attrs.push_back(
      rewriter.getNamedAttr("pad_value", op.getPadValueAttr()));
    attrs.push_back(
      rewriter.getNamedAttr("pad_mode", kls::PaddingModeAttr::get(getContext(), pad_mode)));

  rewriter.replaceOpWithNewOp<tbc::kls::Pool2DOp>(op, outputType, input,
                                                        attrs);
  return success();
  }

}
