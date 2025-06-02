


#include "support/mathutil.h"
#include"support/module.h"
using namespace tbc::ops;

struct MoveReluAheadConcatPattern : public OpRewritePattern<ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op,
                                PatternRewriter &rewriter) const override {
    std::string op_name = op.getOperationName().str();
    auto relu_limit = op.getReluLimit();
    // match relu Op that is following concat Ops
    auto formerOp = op->getOperand(0).getDefiningOp();
    if (!formerOp->hasOneUse() || !isa<ConcatOp>(formerOp)) {
      return failure();
    }

    auto concatOp = cast<ConcatOp>(formerOp);
    int num_inputs = concatOp.getInputs().size();
    rewriter.setInsertionPoint(formerOp);
    for (int i = 0; i < num_inputs; i++) {
      auto inOp = formerOp->getOperand(i).getDefiningOp();

      auto inOp_name = module::getName(inOp).str();
      std::string new_name = inOp_name + "_move_ahead_relu";
      auto nameAttr = rewriter.getStringAttr(new_name);
      auto newOp = rewriter.create<ReluOp>(
          NameLoc::get(nameAttr), formerOp->getOperand(i).getType(),
          ArrayRef<Value>{formerOp->getOperand(i)});
      formerOp->setOperand(i, newOp.getResult());
    }

    // change the concat Op's name to avoid comparison between concat before and after relu
    concatOp->setLoc(NameLoc::get(
        rewriter.getStringAttr(module::getName(formerOp).str() + "_relu")));

    rewriter.replaceOp(op, concatOp);
    return success();
  }
};

void ReluOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.insert<MoveReluAheadConcatPattern>(context);
}
