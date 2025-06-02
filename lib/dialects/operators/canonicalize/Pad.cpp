

#include "support/mathutil.h"
#include "support/module.h"
#include "support/utils.h"
using namespace tbc::ops;

struct TopFusePad : public OpRewritePattern<PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp op,
                                PatternRewriter &rewriter) const override {

    if (dyn_cast<TensorType>(op.getInput().getType()).getShape().size() < 3)
      return failure();

    auto paddings = module::getI64Array(op.getPaddings());
    // without batch or channel padding
    int tensor_dim = paddings.get()->size() / 2;
    for (int i = 0; i < 2; ++i) {
      if (paddings.get()->at(i) != 0 || paddings.get()->at(i + tensor_dim) != 0)
        return failure();
    }
    int pad_dim = tensor_dim - 2;

    // only const pad
    auto pad_mode = op.getMode();
    if ( pad_mode != ::tbc::utils::PaddingMode::constant)
      return failure();

    // check next op, pad_value and pad algo
    double pad_value = cast<FloatAttr>(op->getAttr("val")).getValueAsDouble();
    for (auto nextOp_iter = op->user_begin();
         nextOp_iter != op->user_end(); nextOp_iter++) {
      auto nextOp = *nextOp_iter;
      if (isa<ConvOp>(nextOp)) {
        if (pad_value != 0)
          return failure();
      } else if (isa<MaxPoolOp, AvgPoolOp>(nextOp)) {
        auto nextOp_pad_value =
            cast<IntegerAttr>(nextOp->getAttr("pad_value")).getInt();
        if (pad_value != double(nextOp_pad_value))
          return failure();
        auto nextOp_count_include_pad =
            cast<BoolAttr>(nextOp->getAttr("count_include_pad")).getValue();
        if (!nextOp_count_include_pad)
          return failure();
      } else
        return failure();
    }

    // remove batch padding and channel padding
    std::vector<int64_t> paddings_(pad_dim * 2, 0);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < pad_dim; ++j) {
        paddings_[i * pad_dim + j] = paddings.get()->at(i * tensor_dim + j + 2);
      }
    }

    // check tensor dims and paddings after merged
    for (auto nextOp_iter = op->user_begin();
         nextOp_iter != op->user_end(); nextOp_iter++) {
      auto nextOp = *nextOp_iter;
      auto kernel_shape = dyn_cast<ArrayAttr>(nextOp->getAttr("kernel_shape"));
      if (kernel_shape.size() != pad_dim)
        return failure();
      auto next_paddings =
          module::getI64Array(dyn_cast<ArrayAttr>(nextOp->getAttr("pads")));
      for (int i = 0; i < pad_dim * 2; ++i) {
        // chip limit
        if (next_paddings.get()->at(i) + paddings_[i] > 15)
          return failure();
      }
    }

    // merge paddings
    for (auto nextOp_iter = op->user_begin();
         nextOp_iter != op->user_end(); nextOp_iter++) {
      std::vector<int64_t> new_paddings(pad_dim * 2, 0);
      auto nextOp = *nextOp_iter;
      auto next_paddings =
          module::getI64Array(dyn_cast<ArrayAttr>(nextOp->getAttr("pads")));
      for (int i = 0; i < pad_dim * 2; ++i) {
        new_paddings[i] = next_paddings.get()->at(i) + paddings_[i];
      }
      nextOp->setAttr("pads", rewriter.getI64ArrayAttr(new_paddings));
    }

    // remove the pad Op
    rewriter.replaceOp(op, {op.getInput()});
    return success();
  }
};

void PadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<TopFusePad>(context);
}
