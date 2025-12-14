
#include "conversions/KernelsToHals/npuV1Lowering.h"
#include "dialects/hals/IR/hals.h"
#include "dialects/hals/IR/halsStructs.h"
#include "dialects/kernels/IR/kernels.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "support/module.h"
#include "llvm/ADT/SmallVector.h"
/*
def Hal_Pool2dOp: Hal_Op<"Pool2d"> {
  let summary = "pool2d operator";
  let description = [{
  }];
  let arguments = (ins
    AnyHalTensor:$input,
    I64ArrayAttr:$kernel_size,
    I64ArrayAttr:$strides,
    Hal_PoolModeAttr:$pool_mode,
    I64ArrayAttr:$pads,
    Hal_PadmodeAttr:$pad_mode,
    DefaultValuedAttr<BoolAttr, "false">:$ceil_mode,
    OptionalAttr<Hal_HardwareInfoAttr>:$hw_info
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyHalTensor:$output);
}

*/


/*
class Kernel_PoolOp<string mnemonic, list<Trait> traits = []> : Kernel_Op<mnemonic,
  !listconcat(traits, [])> {
  let summary = "pool operator";
  let description = [{
    This operation performs pooling on the input tensor.
    The pooling operation can be either average or max pooling.
    The kernel size and stride can be specified to control the pooling operation.
  }];
  let arguments = (ins
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
  let hasCanonicalizer = 0;
  let hasVerifier = 0;
  let extraClassDeclaration = [{
    tbc::utils::pool_attr_t parseParam();
  }];
  let results = (outs AnyTensor:$output);
}
*/
namespace tbc::npuv1 {
LogicalResult
Pool2DOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
  auto pool2dOp = cast<tbc::kls::Pool2DOp>(op);
  llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
    auto ori_type =
      cast_or_null<RankedTensorType>(pool2dOp.getOutput().getType());

  auto returnType = typeConverter->convertType(ori_type);
  //copy old attributes
  attrs.push_back(rewriter.getNamedAttr("kernel_size", pool2dOp.getKernelShapeAttr()));
  attrs.push_back(rewriter.getNamedAttr("strides", pool2dOp.getStridesAttr()));
  attrs.push_back(rewriter.getNamedAttr("pads", pool2dOp.getPadsAttr()));
  auto poolmode=hals::PoolModeAttr::get(getContext(), tbc::hals::symbolizePoolMode(tbc::kls::stringifyPoolMode(pool2dOp.getPoolMode())).value());
  attrs.push_back(rewriter.getNamedAttr("pool_mode", poolmode));
  auto padmode=tbc::kls::stringifyPaddingMode(pool2dOp.getPadMode());
  auto hal_pad_mode=hals::PadmodeAttr::get(getContext(), tbc::hals::symbolizePadmode(padmode).value());
  attrs.push_back(rewriter.getNamedAttr("pad_mode", hal_pad_mode));
  attrs.push_back(rewriter.getNamedAttr("ceil_mode", BoolAttr::get(getContext(), false)));
  //create hal pool2d op
  rewriter.replaceOpWithNewOp<tbc::hals::Pool2dOp>(op,returnType, operands, attrs);
  return success();
}

}
