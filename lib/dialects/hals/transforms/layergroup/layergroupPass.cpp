#include "dialects/hals/IR/hals.h"
#include "dialects/hals/transforms/layergroup/base.h"
#include "dialects/hals/transforms/pass.h"
#include "dialects/operators/IR/operator.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "support/module.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <memory>
namespace tbc::hals {

std::vector<LgInfo>
splitLgIrTogroups(llvm::SetVector<mlir::Operation *> &totalops) {
  std::vector<LgInfo> lg_infos;
  LgInfo lg_info;
  for (auto op : totalops) {
    if (mlir::isa<LayerGroupInferInterface>(op)) {
      auto lg_op = mlir::cast<LayerGroupInferInterface>(op);
      if (lg_op.supportGroupTiling()) {
        lg_info.group_ops.insert(op);
      } else {
        // for not support group tiling op, we directly push it to a layergoup
        // alone.
        if (!lg_info.group_ops.empty()) {
          lg_infos.push_back(lg_info);
          lg_info.group_ops.clear();
        }
        lg_info.group_ops.insert(op);
        lg_infos.push_back(lg_info);
        lg_info.group_ops.clear();
      }
    } else {
      // for op without LayerGroupInferInterface just skip
      if (lg_info.group_ops.empty()) {
        continue;
      } else {
        lg_infos.push_back(lg_info);
        lg_info.group_ops.clear();
      }
    }
  }
  if (!lg_info.group_ops.empty()) {
    lg_infos.push_back(lg_info);
  }
  return lg_infos;
}

llvm::LogicalResult initLgInfo(LgInfo &lg_info) {
  for (auto op : lg_info.group_ops) {
    for (auto value : op->getOperands()) {
      if (lg_info.group_ins.contains(value) ||
          lg_info.group_weights.contains(value)) {
        continue;
      }
      if (mlir::isa<mlir::BlockArgument>(value)) {
        LOGE << "value " << value
             << " is block argument, should be considered in layergroup";
        llvm_unreachable("invalid arguments");
      } else {
        auto defineOp = value.getDefiningOp();
        if (mlir::isa<ops::NoneOp>(defineOp)) {
          // placeholder for optional inputs
          continue;
        }
        if (lg_info.group_ops.contains(defineOp)) {
          // if the value is defined by an op in the same layergroup, we can
          // ignore it since it is an intermediate value.
          continue;
        } else {
          if (mlir::isa<hals::WeightOp, hals::PackedWeightGroupOp>(defineOp)) {
            lg_info.group_weights.insert(value);
          } else {
            lg_info.group_ins.insert(value);
          }
        }
      }
    }
    // for  outputs
    for (auto value : op->getResults()) {
      if (mlir::isa<mlir::NoneType>(value.getType())) {
        continue;
      }
      for (auto user : value.getUsers()) {
        if (lg_info.group_ops.contains(user)) {
          // if the user is in the same layergroup, we can ignore it since it
          // is an intermediate value.
          continue;
        } else {
          lg_info.group_outs.insert(value);
          break;
        }
      }
    }
  }
  return llvm::success();
}
hals::LoadOp createLoadOp(mlir::Block * block,const llvm::SetVector<mlir::Operation *>&ops,mlir::MLIRContext *ctx,mlir::Operation *&cur_op, mlir::Value input) {
  mlir::OpBuilder builder(ctx);
  if(cur_op ==nullptr){
    builder.setInsertionPointToStart(block);
  }else{
    builder.setInsertionPointAfter(cur_op);
  }
  auto inputOp = input.getDefiningOp();
  std::string name = "load_";
  if (inputOp != nullptr) {
    name = name + module::getName(inputOp).str();
  } else {
    name = name + std::to_string(cast<BlockArgument>(input).getArgNumber());
  }
  auto loadOp = builder.create<hals::LoadOp>(NameLoc::get(builder.getStringAttr(name)), input.getType(), input);
  cur_op = loadOp.getOperation();
  input.replaceUsesWithIf(loadOp->getResult(0), [&](mlir::OpOperand &operand) {
    auto user = operand.getOwner();
    return ops.contains(user);
  });
  return loadOp;
}
hals::LoadWeightOp createLoadWeightOp( mlir::Block * block,const llvm::SetVector<mlir::Operation *>&ops,mlir::MLIRContext *ctx,mlir::Operation *&cur_op, mlir::Value input) {
   mlir::OpBuilder builder(ctx);
  if(cur_op ==nullptr){
    builder.setInsertionPointToStart(block);
  }else{
    builder.setInsertionPointAfter(cur_op);
  }
  std::string name = "load_weight_";

  name = name + module::getName(input).str();
  
  auto loadOp = builder.create<hals::LoadWeightOp>(NameLoc::get(builder.getStringAttr(name)), input.getType(), input);
  cur_op = loadOp.getOperation();
  input.replaceUsesWithIf(loadOp->getResult(0), [&](mlir::OpOperand &operand) {
    auto user = operand.getOwner();
    return ops.contains(user);
  });
  return loadOp;
}
hals::StoreOp createStoreOp( mlir::Block *block,const llvm::SetVector<mlir::Operation *>&ops,mlir::MLIRContext *ctx,mlir::Operation *&cur_op,
                                  mlir::Value store_val) {
  mlir::OpBuilder builder(ctx);
  if(cur_op ==nullptr){
    builder.setInsertionPointToStart(block);
  }else{
    builder.setInsertionPointAfter(cur_op);
  }
  auto storeOp = builder.create<hals::StoreOp>(module::getLoc(store_val), store_val.getType(), store_val);
  cur_op = storeOp.getOperation();
  return storeOp;
}

llvm::LogicalResult rebuildLayerGroup(LgInfo &lg_info) {
  auto ctx = lg_info.group_ops[0]->getContext();
  llvm::SmallVector<mlir::Type, 8> out_ty;
  llvm::SmallVector<mlir::Location, 4> locs;
  llvm::SmallVector<mlir::Value, 8> operands;
  llvm::DenseSet<mlir::Value> handled_inputs_weights;
  llvm::SmallVector<mlir::Value, 8> origin_outs;
  mlir::Operation *insert_point = nullptr;
  llvm::SmallVector<mlir::Value, 4> stores;
  auto builder = mlir::OpBuilder(ctx);
  builder.setInsertionPointAfter(lg_info.group_ops.back());
  for (auto value : lg_info.group_outs) {
    out_ty.push_back(value.getType());
    locs.push_back(value.getLoc());
    origin_outs.push_back(value);
  }
  for (auto input : lg_info.group_ins) {
    operands.push_back(input);
  }
  auto group_loc = builder.getFusedLoc(locs);
  auto layer_group_op =
      builder.create<hals::GroupOp>(group_loc, out_ty, operands);
  mlir::Block *body = new mlir::Block();
  layer_group_op.getBody().push_back(body);
  for (auto [idx, val] : llvm::enumerate(layer_group_op->getResults())) {
    origin_outs[idx].replaceUsesWithIf(val, [&](mlir::OpOperand &operand) {
      auto user = operand.getOwner();
      return !lg_info.group_ops.contains(user);
    });
  }

  // build loadOp/loadWeightOp/storeOp/normal op
  for (auto &op : lg_info.group_ops) {
    for (auto operand : op->getOperands()) {
      if (handled_inputs_weights.contains(operand)) {
        continue;
      }
      if (lg_info.group_ins.contains(operand)) {
         auto loadOp=createLoadOp(body,lg_info.group_ops,ctx, insert_point, operand);
         handled_inputs_weights.insert(loadOp.getResult());
        
      } else if (lg_info.group_weights.contains(operand)) {
          auto loadWeightOp=createLoadWeightOp(body,lg_info.group_ops,ctx,insert_point, operand);
          handled_inputs_weights.insert(loadWeightOp.getResult());
      }
    }
    // move the original op into layergroup
    op->moveAfter(insert_point);
    insert_point = op;
    // for the results of the op, if it is an output of the layergroup, we need
    // to create store op to store the result to the output value.
    for (auto res : op->getResults()) {
      if (lg_info.group_outs.contains(res)) {
          auto storeOp=createStoreOp(body,lg_info.group_ops,ctx,insert_point, res);
          stores.push_back(storeOp.getResult());
      }
    }
  }
  builder.setInsertionPointAfter(insert_point);
  llvm::SmallVector<mlir::Value, 8> new_stores;
  for (auto &loc : locs) {
    for (auto &s : stores) {
      if (module::getLoc(s) == loc) {
        new_stores.push_back(s);
        break;
      }
    }
  }
  assert(new_stores.size() == stores.size());
  builder.create<hals::YieldOp>(group_loc, new_stores);
  return llvm::success();
}

class LayerGroupPass : public LayerGroupPassBase<LayerGroupPass> {
public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    LayerGroupIR lg_ir;
    // Collect all operations in the function
    funcOp.walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::func::FuncOp, hals::InputOp, hals::ReturnOp,
                    ops::NoneOp, hals::WeightOp, hals::PackedWeightGroupOp>(op))
        return mlir::WalkResult::advance();
      else {
        lg_ir.layerOps.insert(op);
        for (auto value : op->getOperands()) {
          if (mlir::isa<mlir::NoneType>(value.getType())) {
            continue;
          }
          lg_ir.values.insert(value);
        }

        for (auto value : op->getResults()) {
          if (mlir::isa<mlir::NoneType>(value.getType())) {
            continue;
          }
          lg_ir.values.insert(value);
        }
        return mlir::WalkResult::advance();
      }
    });

    // split layer groups to LgInfo
    // currently we build one layergroup  just for test
    lg_ir.lg_infos = splitLgIrTogroups(lg_ir.layerOps);
    for (auto &lg_info : lg_ir.lg_infos) {
      if (llvm::failed(initLgInfo(lg_info))) {
        LOGE << "failed to init lg_info";
        signalPassFailure();
        return;
      }
    }
    // rebuild layergroup base on lgInfo
    for (auto [idx, lg_info] : llvm::enumerate(lg_ir.lg_infos)) {
      if (llvm::failed(rebuildLayerGroup(lg_info))) {
        LOGE << "failed to rebuild layergroup for group " << idx;
        signalPassFailure();
        return;
      }
    }
  }
};

std::unique_ptr<mlir::OperationPass<FuncOp>> createLayerGroupPass() {
  return std::make_unique<LayerGroupPass>();
}
} // namespace tbc::hals
