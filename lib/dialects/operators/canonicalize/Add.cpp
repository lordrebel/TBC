

#include "support/module.h"
#include "mlir/Parser/Parser.h"
#include "dialects/operators/pdll/Canonical/AddPatterns.h.inc"
using namespace tbc::ops;

struct SwapInput : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }
    if (!isa<WeightOp>(
            module::getOriValue(op.getInputs()[0]).getDefiningOp()) &&
        !isa<WeightOp>(
            module::getOriValue(op.getInputs()[1]).getDefiningOp())) {
      return failure();
    }
    auto coeffs = module::getF64Array(op.getCoeff(), 2, 1.0);
    for (auto c : *coeffs) {
      if (c != 1.0) {
        return failure();
      }
    }
    auto lhs = op.getInputs()[0];
    auto rhs = op.getInputs()[1];
    if (isa<WeightOp>(module::getOriValue(lhs).getDefiningOp())) {
      op.setOperand(0, rhs);
      op.setOperand(1, lhs);
      return success();
    } else {
      return failure();
    }
  }
};

struct AddToScale : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }
    if (!isa<WeightOp>(op.getInputs()[1].getDefiningOp())) {
      return failure();
    }
    auto lhs_shape = module::getShape(op.getInputs()[0]);
    auto rhs_shape = module::getShape(op.getInputs()[1]);
    auto output_shape = module::getShape(op.getOutput());
    auto coeffs = module::getF64Array(op.getCoeff(), 2, 1.0);
    for (auto c : *coeffs) {
      if (c != 1.0) {
        return failure();
      }
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32()) {
      return failure();
    }

    if (output_shape.size() < 2 || lhs_shape.size() != rhs_shape.size() ||
        output_shape.size() - rhs_shape.size() > 1) {
      return failure();
    }

    if (rhs_shape[1] != lhs_shape[1]) {
      return failure();
    }

    auto elt_num = module::getNumElements(op.getInputs()[1]);
    if (elt_num != lhs_shape[1]) {
      return failure();
    }

    std::vector<NamedAttribute> attrs;
    std::vector<Value> operands;
    rewriter.setInsertionPoint(op);
    std::vector<float_t> weight_v(elt_num, 1.);

    auto rtype = RankedTensorType::get(rhs_shape.vec(),  module::getElementType(op.getInputs()[0]));
    auto w_scale =
        WeightOp::create(op.getOperation(), "_scale_weight", weight_v, rtype);
    operands.push_back(op.getInputs()[0]);
    operands.push_back(w_scale);
    operands.push_back(op.getInputs()[1]);
    rewriter.replaceOpWithNewOp<ScaleOp>(op, op.getType(), operands, attrs);
    return success();
  }
};

struct AddToAddConst : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }
    auto coeffs = module::getF64Array(op.getCoeff(), 2, 1.0);
    for (auto c : *coeffs) {
      if (c != 1.0) {
        return failure();
      }
    }
    int left_elt_num = module::getNumElements(op.getInputs()[0]);
    int right_elt_num = module::getNumElements(op.getInputs()[1]);

    Value new_input;
    std::shared_ptr<std::vector<float>> const_val;
    bool weight_flag = false;
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32())
      return failure();
    if (left_elt_num == 1) {
      if (auto left_op =
              dyn_cast_or_null<WeightOp>(op.getInputs()[0].getDefiningOp())) {
        weight_flag = true;
        const_val = left_op.read<float>();
      }
      new_input = op.getInputs()[1];
    }
    if (!weight_flag && right_elt_num == 1) {
      if (auto right_op =
              dyn_cast<WeightOp>(op.getInputs()[1].getDefiningOp())) {
        weight_flag = true;
        const_val = right_op.read<float>();
      }
      new_input = op.getInputs()[0];
    } else {
      return failure();
    }
    if (!weight_flag) {
      return failure();
    }
    if (const_val->at(0) == 0.0f) {
      rewriter.replaceOp(op, {new_input});
      return success();
    }
    Type output = op.getOutput().getType();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "const_val", rewriter.getF64FloatAttr(const_val->at(0))));
    rewriter.replaceOpWithNewOp<AddConstOp>(op, output, new_input, attrs);
    return success();
  }
};


void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  populateGeneratedPDLLPatterns(results);
  results.insert<AddToAddConst,AddToScale>(context);
}
