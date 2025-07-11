

#include "support/module.h"
#define MAX_SHAPE_DIMS 8
using namespace tbc::ops;

struct TopReduceTranspose : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;
  TopReduceTranspose(MLIRContext *context, PatternBenefit benefit = 9)
      : OpRewritePattern<ReduceOp>(context, benefit) {}

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {

    auto formerOp = op.getInput().getDefiningOp();
    if (!formerOp->hasOneUse() || !isa<PermuteOp>(formerOp)) {
      return failure();
    }
    auto permuteOp = cast<PermuteOp>(formerOp);
    if (op.getKeepdims()) {
      return failure();
    }
    auto reduce_axes = module::getI64Array(op.getAxes());
    auto permute_order = module::getI64Array(permuteOp.getOrder());
    auto input_shape = module::getShape(op.getInput());
    auto input_dim = input_shape.size();
    // reduce all axes, permute can be erased if not used by others
    if (input_dim == reduce_axes->size()) {
      op->setOperand(0, permuteOp.getInput());
      if (permuteOp.getOutput().getUsers().empty()) {
        rewriter.eraseOp(permuteOp);
      }
      return success();
    }
    std::vector<int64_t> new_axis(reduce_axes->size(), 0);
    for (int i = 0; i < reduce_axes->size(); i++) {
      int axis = reduce_axes->at(i);
      axis = axis < 0 ? (axis + input_dim) : axis;
      assert((axis >= 0 && axis < input_dim) && "0 <= axis < in_dims");
      new_axis[i] = permute_order->at(axis);
    }
    /*
    fuse need satisfied two conditions belowL
    1. for input shape all the transposed axes is closed to each other, in other
    word, we have only one "continuous transposed axes range"
    2. in the transposed range, we have less than one dim after reduce
    */
    int left_count = 0;
    int continuous_transposed_range_num = 0;
    int continuous_transposed_sign = 0;
    for (int i = 0; i < input_dim; i++) {
      if (permute_order->at(i) != i) {
        left_count++;
        if (0 == continuous_transposed_sign) {
          continuous_transposed_range_num++;
          continuous_transposed_sign = 1;
        }
        for (int j = 0; j < reduce_axes->size(); j++) {
          if (new_axis[j] == permute_order->at(i)) {
            left_count--;
            break;
          }
        }
      } else if (1 == continuous_transposed_sign) {
        continuous_transposed_sign = 0;
      }
    }
    if (left_count > 1 || continuous_transposed_range_num > 1) {
      return failure();
    }
    std::sort(new_axis.begin(), new_axis.end());
    op->setAttr("axes", rewriter.getI64ArrayAttr(new_axis));
    op->setOperand(0, permuteOp.getInput());
    if (permuteOp.getOutput().getUsers().empty()) {
      rewriter.eraseOp(permuteOp);
    }
    return success();
  }
};

struct ReduceFusePattern : public OpRewritePattern<ReduceOp> {

  using OpRewritePattern::OpRewritePattern;
  ReduceFusePattern(MLIRContext *context, PatternBenefit benefit = 10)
      : OpRewritePattern<ReduceOp>(context, benefit) {}
  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto formerOpDefine = op.getInput().getDefiningOp();
    if (!isa<ReduceOp>(op) || !isa<ReduceOp>(formerOpDefine)) {
      return failure();
    }
    auto formerOp = cast<ReduceOp>(formerOpDefine);
    if (formerOp.getMode() != op.getMode()) {
      return failure();
    }
    if (formerOp.getKeepdims() != op.getKeepdims()) {
      return failure();
    }
    if (!formerOp->getResult(0).hasOneUse()) {
      return failure();
    }
    auto axis_list_former = module::getI64Array(formerOp.getAxes());
    auto axis_list_current = module::getI64Array(op.getAxes());
    int axis_num_former = axis_list_former->size();
    int axis_num_current = axis_list_current->size();
    auto new_input = formerOp.getInput();

    int mask[MAX_SHAPE_DIMS] = {0};
    int new_axis_num = 0;
    for (int i = 0; i < axis_num_former; i++) {
      mask[axis_list_former->at(i)] = 1;
      new_axis_num += 1;
    }
    for (int i = 0; i < axis_num_current; i++) {
      int offset = 0;
      while (mask[axis_list_current->at(i) + offset]) {
        offset += 1;
      }
      mask[axis_list_current->at(i) + offset] = 1;
      new_axis_num += 1;
    }
    int offset_start = 0;
    while (!mask[offset_start]) {
      offset_start += 1;
    }
    std::vector<int64_t> new_axis(new_axis_num, 0);
    for (int i = 0; i < new_axis_num; i++) {
      int offset_insert = offset_start;
      while (!mask[offset_insert]) {
        offset_insert += 1;
      }
      new_axis[i] = offset_insert;
      offset_start = offset_insert + 1;
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr(new_axis)));
    attrs.push_back(rewriter.getNamedAttr("keepdims", op.getKeepdimsAttr()));
    attrs.push_back(rewriter.getNamedAttr("mode", op.getModeAttr()));
    rewriter.replaceOpWithNewOp<ReduceOp>(op, op.getResult().getType(),
                                          new_input, attrs);
    return success();
  }
};

struct ReduceToReshapePattern : public OpRewritePattern<ReduceOp> {

  using OpRewritePattern::OpRewritePattern;
  ReduceToReshapePattern(MLIRContext *context, PatternBenefit benefit = 10)
      : OpRewritePattern<ReduceOp>(context, benefit) {}
  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto nofInputElts = module::getNumElements(op.getInput());
    auto nofOutputElts = module::getNumElements(op.getOutput());
    if (nofInputElts != nofOutputElts) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, op.getType(), op.getInput(),
                                           std::vector<NamedAttribute>());
    return success();
  }
};

struct ReduceToRPoolPattern : public OpRewritePattern<ReduceOp> {

  using OpRewritePattern::OpRewritePattern;
  ReduceToRPoolPattern(MLIRContext *context, PatternBenefit benefit = 10)
      : OpRewritePattern<ReduceOp>(context, benefit) {}
  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto iShape = module::getShape(op.getInput());
    auto num_dims = iShape.size();
    auto mode = op.getMode().str();
    auto axes = module::getI64Array(op.getAxes());
    if ((mode == "ReduceMean" || mode == "ReduceMax") && num_dims == 4 &&
        axes->size() == 2 && axes->at(0) == num_dims - 2 &&
        axes->at(1) == num_dims - 1) {
      // prepare attrs
      std::vector<NamedAttribute> pool_attrs;
      pool_attrs.emplace_back(rewriter.getNamedAttr(
          "kernel_shape", rewriter.getI64ArrayAttr(
                              {iShape[num_dims - 2], iShape[num_dims - 1]})));
      pool_attrs.emplace_back(
          rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr({1, 1})));
      pool_attrs.emplace_back(rewriter.getNamedAttr(
          "pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));
      pool_attrs.emplace_back(rewriter.getNamedAttr(
          "count_include_pad", rewriter.getBoolAttr(true)));
      pool_attrs.emplace_back(
          rewriter.getNamedAttr("keepdims", op.getKeepdimsAttr()));
      if (mode == "ReduceMean") {
        rewriter.replaceOpWithNewOp<AvgPoolOp>(op, op.getType(), op.getInput(),
                                               pool_attrs);
      } else {
        rewriter.replaceOpWithNewOp<MaxPoolOp>(op, op.getType(), op.getInput(),
                                               pool_attrs);
      }
      return success();
    } else {
      return failure();
    }
  }
};

void ReduceOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<ReduceToReshapePattern, ReduceToRPoolPattern,
                 TopReduceTranspose, ReduceFusePattern>(context);
}
