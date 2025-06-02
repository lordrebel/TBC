

#include "support/mathutil.h"
#include "support/module.h"
using namespace tbc::ops;

// unsqueeze + squeeze && in == out
struct TopFuseSqueeze : public OpRewritePattern<SqueezeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SqueezeOp op,
                                PatternRewriter &rewriter) const override {
    auto in_op = op.getInput().getDefiningOp();
    if (in_op->hasOneUse() && isa<UnsqueezeOp>(in_op)) {
      auto former_op = dyn_cast<UnsqueezeOp>(in_op);
      auto shape0 = module::getShape(op.getOutput());
      auto shape1 = module::getShape(former_op.getInput());
      if (shape0 != shape1) {
      return failure();
      }
      op.getOutput().replaceAllUsesWith(former_op.getInput());
      rewriter.eraseOp(op);
      rewriter.eraseOp(former_op);
      return success();
      }
    return failure();
  }
};

void SqueezeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<TopFuseSqueeze>(context);
}
