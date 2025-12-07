#include "conversions/KernelsToHals/npuV1Lowering.h"
#include "dialects/hals/IR/hals.h"
#include "dialects/hals/IR/halsStructs.h"
#include "dialects/kernels/IR/kernels.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "support/module.h"
#include "llvm/ADT/SmallVector.h"

/*

def Kernel_Conv2dOp : Kernel_Op<"Conv2d"> {
  let summary = "Conv2d operator";
  let description = [{
    The operator for 2d convolution function
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    DefaultValuedAttr<I64Attr, "1">:$group,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<BoolAttr>:$do_winograd,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
     tbc::utils::conv_attr_t parseParam();
  }];
}

*/
/*
def Hal_Conv2dOp: Hal_Op<"Conv2d"> {
  let summary = "conv2d operator";
  let description = [{
  }];
  let arguments = (ins
    AnyHalTensor:$input,
    AnyHalTensor:$weight,
    AnyHalTensorOrNone:$bias,
    AnyHalTensorOrNone:$lut_k,
    AnyHalTensorOrNone:$lut_x,
    AnyHalTensorOrNone:$lut_y,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    I64ArrayAttr:$dilations,
    SI32Attr:$group,
    DefaultValuedAttr<BoolAttr,"false">:$do_winograd,
    Hal_PadmodeAttr:$pad_mode,
    DefaultValuedAttr<BoolAttr, "false">:$fused_lut,
    OptionalAttr<Hal_LutAttr>:$lut_attr,
    OptionalAttr<Hal_HardwareInfoAttr>:$hw_info
  );
}

*/

namespace tbc::npuv1 {
LogicalResult
Conv2dOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
  auto conv2dOp = cast<tbc::kls::Conv2dOp>(op);
  auto ori_type =
      cast_or_null<RankedTensorType>(conv2dOp.getOutput().getType());
  if (!ori_type) {
    llvm::errs() << "Conv2dLowering returnType:" << conv2dOp.getType() << "\n";
    llvm_unreachable("invalid");
  }

  llvm::SmallVector<Value, 4> newOperands(operands);
  auto noneOp = module::getNoneOp(op);
  newOperands.push_back(noneOp->getResult(0));
  newOperands.push_back(noneOp->getResult(0));
  newOperands.push_back(noneOp->getResult(0));

  auto returnType = typeConverter->convertType(ori_type);

  llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("kernel_shape", conv2dOp->getAttr("kernel_shape")));
  attrs.push_back(rewriter.getNamedAttr("pads", conv2dOp->getAttr("pads")));
  attrs.push_back(
      rewriter.getNamedAttr("strides", conv2dOp->getAttr("strides")));
  attrs.push_back(
      rewriter.getNamedAttr("dilations", conv2dOp->getAttr("dilations")));
  attrs.push_back(rewriter.getNamedAttr("group", conv2dOp->getAttr("group")));
  attrs.push_back(
      rewriter.getNamedAttr("do_winograd", conv2dOp->getAttr("do_winograd")));
  attrs.push_back(rewriter.getNamedAttr(
      "pad_mode",
      hals::PadmodeAttr::get(getContext(), ::tbc::hals::Padmode::constant)));
  attrs.push_back(
      rewriter.getNamedAttr("fused_lut", rewriter.getBoolAttr(false)));

  rewriter.replaceOpWithNewOp<tbc::hals::Conv2dOp>(op, returnType, newOperands,
                                                   attrs);
  return success();
}
} // namespace tbc::npuv1
