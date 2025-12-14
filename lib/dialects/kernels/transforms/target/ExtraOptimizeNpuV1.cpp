#include "dialects/kernels/IR/kernels.h"
#include "dialects/kernels/transforms/target/targetRegistry.h"
#include "dialects/operators/IR/operator.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "support/log.h"
#include "support/module.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

namespace tbc::kls {

template <kls::ActiveMode>
llvm::LogicalResult convertActTolut(ActiveOp op, PatternRewriter &rewriter) {
  llvm_unreachable("not inplement yet");
}

template <>
llvm::LogicalResult
convertActTolut<kls::ActiveMode::RELU>(ActiveOp op, PatternRewriter &rewriter) {
  // fake convertion
  if (module::getElementType(op.getOutput()).isF16()) {
    auto lut_x_data = std::vector<uint16_t>(16, 0);
    auto lut_y_data = std::vector<uint16_t>(16, 0);
    auto lut_k_data = std::vector<uint16_t>(16, 0);

    auto lut_x_type =
        RankedTensorType::get({16}, module::getElementType(op.getOutput()));
    auto lut_x_weight =
        ops::WeightOp::create(op, "lut_x", lut_x_data, lut_x_type);
    auto lut_y_type =
        RankedTensorType::get({16}, module::getElementType(op.getOutput()));
    auto lut_y_weight =
        ops::WeightOp::create(op, "lut_y", lut_y_data, lut_y_type);
    auto lut_k_type =
        RankedTensorType::get({16}, module::getElementType(op.getOutput()));
    auto lut_k_weight =
        ops::WeightOp::create(op, "lut_k", lut_k_data, lut_k_type);

    auto inputs = llvm::SmallVector<mlir::Value, 4>(op->getOperands().begin(),
                                                    op->getOperands().end());
    auto attrs = llvm::SmallVector<mlir::NamedAttribute, 4>();
    attrs.push_back(rewriter.getNamedAttr(
        "lut_attr", kls::LutAttrAttr::get(module::getCtx(), 32, 1, 32)));
    inputs.push_back(lut_x_weight);
    inputs.push_back(lut_y_weight);
    inputs.push_back(lut_k_weight);

    rewriter.replaceOpWithNewOp<kls::LutOp>(op, op.getOutput().getType(),
                                            inputs, attrs);

  } else if (module::getElementType(op.getOutput()).isF32()) {
    auto lut_x_data = std::vector<float>(16, 0);
    auto lut_y_data = std::vector<float>(16, 0);
    auto lut_k_data = std::vector<float>(16, 0);

    auto lut_x_type =
        RankedTensorType::get({16}, module::getElementType(op.getOutput()));
    auto lut_x_weight =
        ops::WeightOp::create(op, "lut_x", lut_x_data, lut_x_type);
    auto lut_y_type =
        RankedTensorType::get({16}, module::getElementType(op.getOutput()));
    auto lut_y_weight =
        ops::WeightOp::create(op, "lut_y", lut_y_data, lut_y_type);
    auto lut_k_type =
        RankedTensorType::get({16}, module::getElementType(op.getOutput()));
    auto lut_k_weight =
        ops::WeightOp::create(op, "lut_k", lut_k_data, lut_k_type);

    auto inputs = llvm::SmallVector<mlir::Value, 4>(op->getOperands().begin(),
                                                    op->getOperands().end());
    auto attrs = llvm::SmallVector<mlir::NamedAttribute, 4>();
    attrs.push_back(rewriter.getNamedAttr(
        "lut_attr", kls::LutAttrAttr::get(module::getCtx(), 32, 1, 32)));
    inputs.push_back(lut_x_weight);
    inputs.push_back(lut_y_weight);
    inputs.push_back(lut_k_weight);

    rewriter.replaceOpWithNewOp<kls::LutOp>(op, op.getOutput().getType(),
                                            inputs, attrs);

  } else if (module::getElementType(op.getOutput()).isFloat8E4M3() ||
             module::getElementType(op.getOutput()).isFloat8E5M2()) {
    auto lut_x_data = std::vector<uint8_t>(16, 0);
    auto lut_y_data = std::vector<uint8_t>(16, 0);
    auto lut_k_data = std::vector<uint8_t>(16, 0);

    auto lut_x_type =
        RankedTensorType::get({16}, module::getElementType(op.getOutput()));
    auto lut_x_weight =
        ops::WeightOp::create(op, "lut_x", lut_x_data, lut_x_type);
    auto lut_y_type =
        RankedTensorType::get({16}, module::getElementType(op.getOutput()));
    auto lut_y_weight =
        ops::WeightOp::create(op, "lut_y", lut_y_data, lut_y_type);
    auto lut_k_type =
        RankedTensorType::get({16}, module::getElementType(op.getOutput()));
    auto lut_k_weight =
        ops::WeightOp::create(op, "lut_k", lut_k_data, lut_k_type);

    auto inputs = llvm::SmallVector<mlir::Value, 4>(op->getOperands().begin(),
                                                    op->getOperands().end());
    auto attrs = llvm::SmallVector<mlir::NamedAttribute, 4>();
    attrs.push_back(rewriter.getNamedAttr(
        "lut_attr", kls::LutAttrAttr::get(module::getCtx(), 32, 1, 32)));
    inputs.push_back(lut_x_weight);
    inputs.push_back(lut_y_weight);
    inputs.push_back(lut_k_weight);

    rewriter.replaceOpWithNewOp<kls::LutOp>(op, op.getOutput().getType(),
                                            inputs, attrs);

  } else {
    LOGE << "not support data type:" << op.getOutput().getType() << "\n";
    llvm_unreachable("not support data type");
  }
  return success();
}

struct ActiveToLut : public OpRewritePattern<ActiveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ActiveOp op,
                                PatternRewriter &rewriter) const override {
    // a fake convertion to convert act op to lut
    if (op.getMode() == kls::ActiveMode::RELU) {
      return convertActTolut<kls::ActiveMode::RELU>(op, rewriter);
    } else {
      llvm_unreachable("not support mode");
    }
    return failure();
  }
};

class ExtraOptimizeNpuV1Pass
    : public ExtraOptimizeNpuV1PassBase<ExtraOptimizeNpuV1Pass> {
public:
  ExtraOptimizeNpuV1Pass() {}
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ActiveToLut>(&getContext());

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::OperationPass<ModuleOp>> createExtraOptimizeNpuV1Pass() {
  return std::make_unique<ExtraOptimizeNpuV1Pass>();
}
} // namespace tbc::kls
