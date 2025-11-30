#pragma once
#include "conversions/conversion.h"
#include "dialects/operators/IR/operator.h"
template <typename OpTy> class OpLowering : public mlir::OpRewritePattern<OpTy> {
public:
  using mlir::OpRewritePattern<OpTy>::OpRewritePattern;

  llvm::LogicalResult matchAndRewrite(OpTy opTy,
                                mlir::PatternRewriter &rewriter) const override{
    llvm_unreachable("Not Implemented");

    }
   
    
};
#define OPLOWERING(OpTy)                                               \
    class OpTy##Lowering : public OpLowering<tbc::ops::OpTy> {           \
    public:                                                             \
        using OpLowering<tbc::ops::OpTy>::OpLowering;                     \
                                                                        \
        llvm::LogicalResult matchAndRewrite(tbc::ops::OpTy op,            \
                                    mlir::PatternRewriter &rewriter) const override; \
    };
namespace tbc::ops {
    OPLOWERING(AddOp);
    OPLOWERING(AddConstOp);
    OPLOWERING(MulOp);
    OPLOWERING(MulConstOp);
    OPLOWERING(SubOp);
    OPLOWERING(SubConstOp);
    OPLOWERING(DivOp);
    OPLOWERING(ReluOp);
    OPLOWERING(ActivationLutOp);
    OPLOWERING(ConcatOp);
    OPLOWERING(ConvOp);
    OPLOWERING(PadOp);
    OPLOWERING(MaxPoolOp);
void populateOperatorsToKernelsConversionPatterns(
    mlir::RewritePatternSet &patterns);
} // namespace tbc::ops
