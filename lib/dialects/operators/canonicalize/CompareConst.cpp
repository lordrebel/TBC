/*
 * (C) Copyright 2025, Imvision Co., Ltd
 * This file is classified as confidential level C2 within Imvision
 * @Date: 2025-06-03 11:45:51
 * Change Logs:
 * 
 * Date           Author         Notes
 * ${now_date}          wangjiahao          initialize 
 */


#include "support/module.h"


using namespace tbc::ops;

struct CompareConstWhereToMinConst : public OpRewritePattern<CompareConstOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CompareConstOp op,
                                PatternRewriter &rewriter) const override {

    auto compare_c_v = op.getConstVal().convertToDouble();
    if (op.getMode().str() == "Less" && op.getResult().hasOneUse() && isa<WhereOp>(*op.getResult().getUsers().begin())) {
      auto where_op = dyn_cast<WhereOp>(*op.getResult().getUsers().begin());
      if (where_op.getXIsConst()) {
        auto where_c_v = where_op.getXConstVal().convertToDouble();
        if (compare_c_v == where_c_v) {
          std::vector<NamedAttribute> attrs;
          attrs.push_back(rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(compare_c_v)));
          where_op.replaceAllUsesWith(op.getOutput());
          rewriter.replaceOpWithNewOp<MinConstOp>(op, op.getOutput().getType(), op.getInput(), attrs);
        }
      }
    }
    return success();
  }
};

void CompareConstOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<CompareConstWhereToMinConst>(context);
}
