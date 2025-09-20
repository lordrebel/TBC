#include "conversions/OperatorsToKernels/opLowering.h"
#include "dialects/kernels/IR/kernels.h"
#include "mlir/IR/Attributes.h"

using namespace mlir;
using namespace llvm;
namespace tbc::ops{
  LogicalResult AddConstOpLowering::matchAndRewrite(tbc::ops::AddConstOp op,
                                PatternRewriter &rewriter) const {
    auto input=op.getInput();
    auto constant=op.getConstVal();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("const_val", rewriter.getF32FloatAttr(constant.convertToFloat())));
    attrs.push_back(rewriter.getNamedAttr("mode", rewriter.getStringAttr("Add")));
    auto outputType=op.getOutput().getType();
    rewriter.replaceOpWithNewOp<tbc::kls::EltWiseConstOp>(op, outputType, input, attrs);
    return success();
  }
}
