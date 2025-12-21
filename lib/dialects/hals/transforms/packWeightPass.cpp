#include "dialects/hals/IR/hals.h"
#include "dialects/hals/IR/halsStructs.h"
#include "dialects/hals/transforms/pass.h"
#include "interfaces/packweightInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "support/module.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace tbc {
namespace hals {
static void createWeightGroupOp(PackWeightInterface op) {
  llvm::SmallVector<hals::WeightOp, 4> weightOps;
  llvm::SmallVector<mlir::Value, 4> weight_outputs;
  llvm::SmallVector<mlir::Location, 4> all_locs;
  llvm::SmallVector<mlir::Type, 4> all_retTypes;
  for (auto val : op->getOperands()) { // TODO maybe ensure value is one use?
    if (auto weightOp = val.getDefiningOp<hals::WeightOp>()) {
      weightOps.push_back(weightOp);
      weight_outputs.push_back(val);
      all_locs.push_back(weightOp->getLoc());
      all_retTypes.push_back(val.getType());
    }
  }
  if (weightOps.size() == 0) {
    LOGW << "no weightOp found in op: %s\n",
        op->getName().getStringRef().data();
    return;
  }
  OpBuilder builder(op->getContext());
  builder.setInsertionPoint(op);
  auto group_loc = builder.getFusedLoc(all_locs);
  auto group_op =
      builder.create<hals::PackedWeightGroupOp>(group_loc, all_retTypes);
  auto block = builder.createBlock(&(group_op.getBody()));
  builder.setInsertionPointToStart(block);
  auto groupReturnOp = builder.create<hals::ReturnOp>(
      mlir::UnknownLoc::get(op->getContext()), weight_outputs);
  for (int i = weightOps.size() - 1; i >= 0; i--) {
    if (i == weightOps.size() - 1) {
      weightOps[i]->moveBefore(groupReturnOp);
    } else {
      weightOps[i]->moveBefore(weightOps[i + 1]);
    }
  }

  for (auto [idx, val] : llvm::enumerate(group_op.getOutputs())) {
    weight_outputs[idx].replaceUsesWithIf(val, [&](OpOperand &operand) {
      Operation *user = operand.getOwner();
      return !isa<hals::PackedWeightGroupOp>(user->getParentOp());
    });
  }
}
class PackWeightPass : public PackWeightsPassBase<PackWeightPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    llvm::SmallVector<PackWeightInterface, 10> ops;
    funcOp.walk([&](PackWeightInterface op) {
      bool valid_op = true;
      for (auto arg : op->getOperands()) {
        if (arg.getDefiningOp<hals::WeightOp>() && !arg.hasOneUse()) {
          LOGE << "ERROR: PackWeightInterface with multiple uses:"
               << op->getName().getStringRef().data();
          valid_op = false;
          break;
        }
      }
      if (valid_op) {
        ops.push_back(op);
      }
    });
    for (auto op : ops) {
      createWeightGroupOp(op);
    }
    // assigin weight address here
    // TODO
  }
};

std::unique_ptr<mlir::OperationPass<FuncOp>> createPackWeightsPass() {
  return std::make_unique<PackWeightPass>();
}
} // namespace hals
} // namespace tbc
