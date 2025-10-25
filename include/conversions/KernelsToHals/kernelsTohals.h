
#pragma once
#include "conversions/conversion.h"
// namespace mlir

namespace mlir {
#define GEN_PASS_DECL
#define GEN_PASS_DEF_CONVERTKERNELSTOHALS
#include "conversions/Pass.h.inc"
} //

#define KernelLowering(Op,benifit) \
class Op##Lowering : public mlir::ConversionPattern { \
public: \
explicit Op##Lowering(TypeConverter &typeConverter,mlir::MLIRContext *ctx) \
      : ConversionPattern(typeConverter, Op::getOperationName(), benifit, ctx) {} \
  LogicalResult \
  matchAndRewrite(Operation *op, ArrayRef<Value> operands, \
                  ConversionPatternRewriter &rewriter) const final; \
};  \

namespace tbc {
  struct ConvertKernelsToHals : public mlir::impl::ConvertKernelsToHalsBase<ConvertKernelsToHals> {

public:
  void runOnOperation() override;

};
}
