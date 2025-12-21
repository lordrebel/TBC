#include "dialects/hals/IR/hals.h"
#include "dialects/hals/IR/halsStructs.h"
#include "dialects/hals/transforms/pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "support/module.h"

namespace tbc {
namespace hals {

class AssginTensorInfoForInputOp
    : public mlir::OpRewritePattern<hals::InputOp> {
public:
  using mlir::OpRewritePattern<hals::InputOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hals::InputOp op,
                                PatternRewriter &rewriter) const override {

    HalTensorType oldTy = cast<HalTensorType>(op.getOutput().getType());
    auto params = oldTy.parse_params();

    if (params.memory_space == hals::MemorySpace::DDR &&
        params.kind == hals::TensorKind::IO &&
        params.layout == hals::StorageLayout::nchw) {
      return failure();
    }
    params.memory_space = hals::MemorySpace::DDR;
    params.kind = hals::TensorKind::IO;
    params.layout = hals::StorageLayout::nchw;
    HalTensorType newTy = HalTensorType::get(getContext(), params);
    op.getOutput().setType(newTy);
    return success();
  }
};

class AssginTensorInfoForWeightOp
    : public mlir::OpRewritePattern<hals::WeightOp> {
public:
  using mlir::OpRewritePattern<hals::WeightOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(hals::WeightOp op,
                                PatternRewriter &rewriter) const override {

    HalTensorType oldTy = cast<HalTensorType>(op.getOutput().getType());
    auto params = oldTy.parse_params();

    if (params.memory_space == hals::MemorySpace::DDR &&
        params.kind == hals::TensorKind::WEIGHT &&
        params.layout == hals::StorageLayout::npu_format) {
      return failure();
    }
    params.memory_space = hals::MemorySpace::DDR;
    params.kind = hals::TensorKind::WEIGHT;
    params.layout = hals::StorageLayout::npu_format;
    HalTensorType newTy = HalTensorType::get(getContext(), params);
    op.getOutput().setType(newTy);
    return success();
  }
};


template<typename OpTy> 
class AssginTensorInfoForOp:
public mlir::OpRewritePattern<OpTy> {
public:
  using mlir::OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    HalTensorType oldTy = cast<HalTensorType>(op.getOutput().getType());
    auto params = oldTy.parse_params();

    if (params.memory_space != hals::MemorySpace::NOT_SET &&
        params.kind != hals::TensorKind::NOT_SET &&
        params.layout == hals::StorageLayout::npu_format) {
      return failure();
    }

    params.memory_space = hals::MemorySpace::L1;
    params.kind = hals::TensorKind::MID;
    params.layout = hals::StorageLayout::npu_format;
    HalTensorType newTy = HalTensorType::get(rewriter.getContext(), params);
    op.getOutput().setType(newTy);
    return success();
  }
};
class AssginTensorInfosPass
    : public AssginTensorInfosBase<AssginTensorInfosPass> {
public:
  AssginTensorInfosPass() {}
  void runOnOperation() override {
    mlir::GreedyRewriteConfig config;
    config.maxIterations = 10;
    config.useTopDownTraversal = true;
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AssginTensorInfoForInputOp>(&getContext());
    patterns.add<AssginTensorInfoForWeightOp>(&getContext());
    
    patterns.add<AssginTensorInfoForOp<hals::Conv2dOp>>(&getContext());
    patterns.add<AssginTensorInfoForOp<hals::Pool2dOp>>(&getContext());
    patterns.add<AssginTensorInfoForOp<hals::EltwiseOp>>(&getContext());
    patterns.add<AssginTensorInfoForOp<hals::EltwiseConstOp>>(&getContext());
    patterns.add<AssginTensorInfoForOp<hals::LutOp>>(&getContext());
    patterns.add<AssginTensorInfoForOp<hals::ScaleConstOp>>(&getContext());
    patterns.add<AssginTensorInfoForOp<hals::ScaleOp>>(&getContext());
    if (mlir::failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }

    //update function type
    getOperation().walk([&](mlir::func::FuncOp funcOp) {
      auto oldTy=funcOp.getFunctionType();
      llvm::SmallVector<mlir::Type, 4> newInputs;
      llvm::SmallVector<mlir::Type, 4> newOutputs;
      for (auto inTy : oldTy.getInputs()) {
        if (auto halTy = dyn_cast<HalTensorType>(inTy)) {
          auto params = halTy.parse_params();
          params.layout=hals::StorageLayout::nchw;
          params.memory_space=hals::MemorySpace::DDR;
          params.kind=hals::TensorKind::IO;
          params.addr=-1;
          HalTensorType newHalTy = HalTensorType::get(funcOp->getContext(), params);
          newInputs.push_back(newHalTy);
        }else{
          newInputs.push_back(inTy);
        }
      }

      for(auto val:funcOp.getBody().getBlocks().front().getTerminator()->getOperands()){
        newOutputs.push_back(val.getType());
      }
      auto newFuncTy = mlir::FunctionType::get(
          funcOp->getContext(), newInputs, newOutputs);
      auto &entryBlock = funcOp.getBody().front();
      for (unsigned i = 0; i < newInputs.size(); ++i) {
        entryBlock.getArgument(i).setType(newInputs[i]);
      }
      funcOp.setType(newFuncTy);
    });
  }
};

std::unique_ptr<mlir::OperationPass<ModuleOp>> createAssginTensorInfosPass() {
  return std::make_unique<AssginTensorInfosPass>();
}

} // namespace hals
} // namespace tbc
