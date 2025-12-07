#include "dialects/kernels/IR/kernels.h"
#include "dialects/kernels/transforms/target/targetRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace tbc::kls{

struct ActiveToLut : public OpRewritePattern<ActiveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ActiveOp op,
                                PatternRewriter &rewriter) const override {
//TODO: Implement the conversion from ActiveOp to LutOp
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

      if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
              getOperation(), std::move(patterns)))) {
        signalPassFailure();
      }

    }
  };

  std::unique_ptr<mlir::OperationPass<ModuleOp>> createExtraOptimizeNpuV1Pass() {
    return std::make_unique<ExtraOptimizeNpuV1Pass>();
  }
}

