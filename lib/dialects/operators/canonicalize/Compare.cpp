

#include "support/module.h"


using namespace tbc::ops;

struct CompareToCompareConst : public OpRewritePattern<CompareOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CompareOp op,
                                PatternRewriter &rewriter) const override {

    auto right_shape = dyn_cast<TensorType>(op.getRhs().getType()).getShape();
    int right_elt_num = 1;
    for (int i = 0; i < right_shape.size(); ++i)
      right_elt_num *= right_shape[i];
    if (right_elt_num > 1)
      return failure();

    std::shared_ptr<std::vector<float>> const_val;
    if (auto right_op = dyn_cast_or_null<WeightOp>(op.getRhs().getDefiningOp())) {
      const_val = right_op.read<float>();
    } else {
      return failure();
    }

    Type output = op.getOutput().getType();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("mode", op.getModeAttr()));
    attrs.push_back(rewriter.getNamedAttr(
        "const_val", rewriter.getF64FloatAttr(const_val->at(0))));
    attrs.push_back(rewriter.getNamedAttr("inversed", rewriter.getBoolAttr(0)));
    rewriter.replaceOpWithNewOp<CompareConstOp>(op, output, op.getLhs(), attrs);
    return success();
  }
};

void CompareOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<CompareToCompareConst>(context);
}
