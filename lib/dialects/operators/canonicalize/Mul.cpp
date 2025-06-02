

#include "support/mathutil.h"
#include "support/module.h"
using namespace tbc::ops;

struct MulToSiLU : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {

    auto in0_op = op.getInputs()[0].getDefiningOp();
    auto in1_op = op.getInputs()[1].getDefiningOp();
    Value in_value;
    SigmoidOp sigmoid_op = dyn_cast<SigmoidOp>(in1_op);
    if (sigmoid_op && sigmoid_op.getInput().getDefiningOp() == in0_op &&
        sigmoid_op->hasOneUse()) {
      in_value = op.getInputs()[0];
    } else if ((sigmoid_op = dyn_cast<SigmoidOp>(in0_op)) &&
               sigmoid_op.getInput().getDefiningOp() == in1_op &&
               sigmoid_op->hasOneUse()) {
      in_value = op.getInputs()[1];
    } else {
      return failure();
    }
    std::vector<NamedAttribute> attrs;
    rewriter.replaceOpWithNewOp<SiLUOp>(op, op.getResult().getType(),
                                        ValueRange{in_value}, attrs);
    return success();
  }
};

/**
 * Weight[1] \
 *            Mul =>  Any -> MulConst(const=WeightData)
 * Any       /
 *
 */
struct MulToMulConst : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getInputs().size() != 2) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32()) {
      return failure();
    }

    int is_const[2];

    is_const[0] = module::getNumElements(op.getInputs()[0]) == 1;
    is_const[1] = module::getNumElements(op.getInputs()[1]) == 1;
    if (!is_const[0] && !is_const[1]) {
      return failure();
    }

    Value new_input;
    std::shared_ptr<std::vector<float>> const_val;
    int weight_index = -1;

    for (int i = 0; i < 2; i++) {
      if (!is_const[i]) {
        continue;
      }
      if (auto weight_op =
              dyn_cast<WeightOp>(op.getInputs()[i].getDefiningOp())) {
        const_val = weight_op.read<float>();
        weight_index = i;
        new_input = op.getInputs()[1 - i]; // take another operand as new input
        break;
      }
    }

    if (weight_index == -1) {
      return failure();
    }

    if (const_val->at(0) == 1.0f) {
      // erase mul
      rewriter.replaceOp(op, {new_input});
      return success();
    }
    Type output = new_input.getType();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "const_val", rewriter.getF64FloatAttr(const_val->at(0))));
    rewriter.replaceOpWithNewOp<MulConstOp>(op, output, new_input, attrs);
    return success();
  }
};

/**
 * ConstantFill \
 *            Mul =>  Any -> MulConst(const=WeightData)
 * Any       /
 *
 */
struct MulToMulConst2 : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getInputs().size() != 2) {
      return failure();
    }
    auto stype = module::getStorageType(op.getOutput());
    if (!stype.isF32()) {
      return failure();
    }
    auto op0 = op.getInputs()[0].getDefiningOp();
    auto op1 = op.getInputs()[1].getDefiningOp();
    Operation *const_op = nullptr;
    Operation *input_op = nullptr;
    if (isa<ConstantFillOp>(op0)) {
      const_op = op0;
      input_op = op1;
    } else if (isa<ConstantFillOp>(op1)) {
      const_op = op1;
      input_op = op0;
    } else {
      return failure();
    }
    auto new_input = input_op->getResult(0);
    auto constOp = cast<ConstantFillOp>(const_op);
    auto in_shape = module::getShape(new_input);
    auto c_shape = module::getShape(constOp.getOutput());
    if (module::getNumElements(constOp.getOutput()) == 1) {
    } else if (in_shape.size() == c_shape.size()) {
      for (auto it : llvm::zip(in_shape, c_shape)) {
        if (std::get<0>(it) < std::get<1>(it)) {
          // shape broadcast
          return failure();
        }
      }
    } else {
      return failure();
    }
    auto const_val = constOp.getValue().convertToDouble();
    if (const_val == 1.0) {
      rewriter.replaceOp(op, {new_input});
      return success();
    }
    Type otype = op.getOutput().getType();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("const_val",
                                          rewriter.getF64FloatAttr(const_val)));
    rewriter.replaceOpWithNewOp<MulConstOp>(op, otype, new_input, attrs);
    return success();
  }
};

struct MulToScale : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }

    // check shape
    auto left_shape =
        dyn_cast<TensorType>(op.getInputs()[0].getType()).getShape();
    auto right_shape =
       dyn_cast<TensorType>( op.getInputs()[1].getType()).getShape();
    if (!(left_shape.size() == 4 && right_shape.size() == 4))
      return failure();
    if (left_shape[1] != right_shape[1])
      return failure();
    int left_elt_num = 1, right_elt_num = 1;
    for (int i = 0; i < left_shape.size(); ++i)
      left_elt_num *= left_shape[i];
    for (int i = 0; i < right_shape.size(); ++i)
      right_elt_num *= right_shape[i];
    if (left_elt_num != left_shape[1] && right_elt_num != right_shape[1])
      return failure();

    // Y = X * S + B
    Value X, S;
    if (left_elt_num == left_shape[1]) {
      X = op.getInputs()[1];
      S = op.getInputs()[0];
    } else if (right_elt_num == right_shape[1]) {
      X = op.getInputs()[0];
      S = op.getInputs()[1];
    } else {
      assert(0);
    }

    std::vector<float> scale(left_shape[1]);
    if (auto scale_ = dyn_cast<WeightOp>(S.getDefiningOp()))
      scale = *(scale_.read<float>());
    else
      return failure();

    auto scale_type =
        RankedTensorType::get({left_shape[1]}, rewriter.getF32Type());
    auto S_ = WeightOp::create(op, "scale", scale, scale_type);
    std::vector<float> bias(left_shape[1], 0);
    auto bias_type =
        RankedTensorType::get({left_shape[1]}, rewriter.getF32Type());
    auto B = WeightOp::create(op, "bias", bias, bias_type);
    std::vector<NamedAttribute> attrs;

    rewriter.replaceOpWithNewOp<ScaleOp>(op, op.getOutput().getType(),
                                         ValueRange{X, S_, B}, attrs);
    return success();
  }
};


struct MergeGelu : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {

    if (!op.getResult().hasOneUse())
      return failure();
    MulConstOp mulconst_op = NULL;
    AddConstOp addconst_op = NULL;
    for (auto in : op.getInputs()) {
      if (auto weight_op = dyn_cast<WeightOp>(in.getDefiningOp()))
        return failure();
      else if ((addconst_op = dyn_cast<AddConstOp>(in.getDefiningOp())))
        continue;
      else if ((mulconst_op = dyn_cast<MulConstOp>(in.getDefiningOp())))
        continue;
      else
        return failure();
    }
    if (mulconst_op == NULL || addconst_op == NULL)
      return failure();
    if (!mulconst_op.getResult().hasOneUse() ||
        !addconst_op.getResult().hasOneUse())
      return failure();
    if (fabs(mulconst_op.getConstVal().convertToDouble() - 0.5) > 1e-4)
      return failure();
    if (fabs(addconst_op.getConstVal().convertToDouble() - 1.0) > 1e-4)
      return failure();
    ErfOp erf_op = NULL;
    erf_op = dyn_cast<ErfOp>(addconst_op.getInput().getDefiningOp());
    if (erf_op == NULL)
      return failure();
    if (!erf_op.getResult().hasOneUse())
      return failure();
    MulConstOp mulconst_op1 = NULL;
    mulconst_op1 = dyn_cast<MulConstOp>(erf_op.getInput().getDefiningOp());
    if (mulconst_op1 == NULL)
      return failure();
    if (fabs(mulconst_op1.getConstVal().convertToDouble() -
             0.70710676908493042f) > 1e-4)
      return failure();
    if (mulconst_op1.getInput().getDefiningOp() !=
        mulconst_op.getInput().getDefiningOp())
      return failure();
    rewriter.replaceOpWithNewOp<GELUOp>(op, op.getResult().getType(),
                                        ValueRange{mulconst_op.getInput()});
    return success();
  }
};

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<MulToSiLU, MulToMulConst, MulToMulConst2, MulToScale,
                 MergeGelu>(context);
}
