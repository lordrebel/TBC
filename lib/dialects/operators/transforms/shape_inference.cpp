//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "dialects/operators/transforms/pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "interfaces/shapeInfer_interface.h"
#include"support/module.h"
#include <llvm/Support/Debug.h>
#define DEBUG_TYPE "shape_infer"

using namespace llvm;
using namespace mlir;
namespace tbc {
namespace ops {

class UnTupleFusePattern : public OpRewritePattern<UnTupleOp> {
public:
  using OpRewritePattern<UnTupleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnTupleOp op,
                                PatternRewriter &rewriter) const override {
    auto outs = op.getOutputs();
    auto ins = op.getInputs();
    if (outs.size() != ins.size()) {
      return failure();
    }
    for (auto it : llvm::zip(ins, outs)) {
      auto in = std::get<0>(it);
      auto out = std::get<1>(it);
      auto loc = module::getLoc(out);
      out.replaceAllUsesWith(in);
      module::setLoc(in, loc);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class TupleFusePattern : public OpRewritePattern<TupleOp> {
public:
  using OpRewritePattern<TupleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TupleOp op,
                                PatternRewriter &rewriter) const override {
    auto out = op.getOutput();
    for (auto user : op->getUsers()) {
      std::vector<Value> operands;
      for (auto opd : user->getOperands()) {
        if (opd == out) {
          for (auto v : op.getOperands()) {
            operands.push_back(v);
          }
        } else {
          operands.push_back(opd);
        }
      }
      user->setOperands(operands);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class CopyMultiUseWeight : public OpRewritePattern<WeightOp> {
public:
  using OpRewritePattern<WeightOp>::OpRewritePattern;

  int getOperandIndex(Operation *op, Value operand) const {
    int n = op->getNumOperands();
    for (int i = 0; i < n; i++) {
      if (operand == op->getOperand(i)) {
        return i;
      }
    }
    llvm_unreachable("operand not found");
    return -1;
  }

  LogicalResult matchAndRewrite(WeightOp op,
                                PatternRewriter &rewriter) const override {
    std::vector<Operation *> users(op->user_begin(), op->user_end());
    if (users.size() <= 1) {
      return failure();
    }
    int idx = 0;
    for (auto user : users) {
      int operand_index = getOperandIndex(user, op.getOutput());
      auto new_weight = op.clone(std::to_string(idx));
      user->setOperand(operand_index, new_weight);
      idx++;
    }
    rewriter.eraseOp(op);
    return success();
  }
};



class ShapeInferPass : public ShapeInferBase<ShapeInferPass> {
public:
  ShapeInferPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    auto ctx = &getContext();
    // Before shape infer
    RewritePatternSet patterns(ctx);
    patterns.add<TupleFusePattern>(ctx);
    applyPatternsAndFoldGreedily(mOp, std::move(patterns));
    patterns.clear();
    patterns.add<UnTupleFusePattern>(ctx);
    applyPatternsAndFoldGreedily(mOp, std::move(patterns));
    patterns.clear();
    patterns.add<CopyMultiUseWeight>(ctx);
    applyPatternsAndFoldGreedily(mOp, std::move(patterns));
    // Do shape infer
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](ShapeInferInterface op) {
        LLVM_DEBUG(llvm::dbgs() << "shape infer: " << op << "\n";);
        op.shape_inference();
        if (false == removeIfNoUse(op)) {
          removeIfNoUse(op);
        }
      });
    }
    module::updateModuleTypes();
  }

private:
  bool removeIfNoUse(Operation *op) {
    // if the op is in the region of other op, don't do removeIfNoUse
    if (op->getUsers().empty()
        && !isa< ops::LoopOp, ops::IfOp>(
                                op->getBlock()->getParentOp())) {
      op->erase();
      return true;
    }
    return false;
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createShapeInferPass() {
  return std::make_unique<ShapeInferPass>();
}
} // namespace top
} // namespace tpu_mlir
