


#include "support/mathutil.h"

using namespace tbc::ops;
using namespace mlir ;
struct PowToBinary : public OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter &rewriter) const override {
    auto exp = op.getExponent().convertToDouble();
    std::vector<NamedAttribute> attrs;
    if (exp == 2) {
      rewriter.replaceOpWithNewOp<MulOp>(op, op.getOutput().getType(),
                                         ValueRange{op.getInput(), op.getInput()},
                                         attrs);
      success();
    } else if (exp == 0.5) {
      rewriter.replaceOpWithNewOp<SqrtOp>(op, op.getOutput().getType(),
                                         ValueRange{op.getInput()},
                                         attrs);
      success();
    }
    return failure();
  }
};

void PowOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<PowToBinary>(context);
}
