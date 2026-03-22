#include "dialects/kernels/transforms/pass.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <queue>

using namespace mlir;
using namespace llvm;

mlir::Operation *getAncestorInBlock(mlir::Operation *op,
                                    mlir::Block *targetBlock) {
  while (op && op->getBlock() != targetBlock) {
    op = op->getParentOp();
  }
  return op;
}
int64_t getMaxPath(mlir::Operation *op,
                   mlir::DenseMap<mlir::Operation *, int64_t> &record) {
  if (record.count(op)) {
    return record[op];
  }
  if (op->getNumOperands() == 0) {
    record[op] = 1;
    return 1;
  }

  int64_t max_path = 0;
  for (auto operand : op->getOperands()) {
    if (isa<BlockArgument>(operand) ||
        operand.getDefiningOp()->getBlock() != op->getBlock()) {
      continue;
    }
    int64_t path = getMaxPath(operand.getDefiningOp(), record);
    if (path > max_path) {
      max_path = path;
    }
  }
  record[op] = max_path + 1;
  return record[op];
}

namespace tbc::kls {
LogicalResult insertOp(mlir::Operation *cur_op, mlir::Block &block,
                       mlir::MLIRContext *ctx, mlir::Operation *&orderd_op) {
  mlir::OpBuilder builder(ctx);
  if (orderd_op == nullptr) {
    builder.setInsertionPointToStart(&block);
  } else {
    builder.setInsertionPointAfter(orderd_op);
  }
  if (orderd_op == nullptr) {
    if (&(block.front()) == cur_op) {
      orderd_op = cur_op;
      return LogicalResult::success();
    } else {
      cur_op->moveAfter(&(block.front()));
      orderd_op = cur_op;
      return LogicalResult::success();
    }
  } else {
    cur_op->moveAfter(orderd_op);
    orderd_op = cur_op;
    return LogicalResult::success();
  }
  return LogicalResult::success();
}
LogicalResult reorderOpsKahn(mlir::Block &block, mlir::MLIRContext *ctx) {
  llvm::DenseMap<mlir::Operation *, int> indegree;
  llvm::DenseMap<mlir::Operation *, llvm::SmallPtrSet<mlir::Operation *, 4>>
      adjList;
  auto terminator =
      block.mightHaveTerminator() ? block.getTerminator() : nullptr;
  // init indegree and adj_list
  for (auto op = block.begin(); op != block.end(); ++op) {
    if (&*op == terminator)
      continue;
    indegree[&*op] = 0;
  }

  for (auto &op : block) {
    if (&op == terminator)
      continue;
    for (mlir::Value res : op.getResults()) {
      for (auto &use : res.getUses()) {
        auto user_op = use.getOwner();
        mlir::Operation *userAncestor = getAncestorInBlock(user_op, &block);
        if (userAncestor && userAncestor != &op && userAncestor != terminator) {
          if (adjList[&op].insert(userAncestor).second) {
            indegree[userAncestor]++;
          }
        }
      }
    }
  }

  std::queue<mlir::Operation *> que;
  for (mlir::Operation &op : block) {
    if (&op == terminator)
      continue;
    if (indegree[&op] == 0) {
      que.push(&op);
    }
  }
  llvm::SmallVector<mlir::Operation *> sortedOps;
  while (!que.empty()) {
    auto op = que.front();
    que.pop();
    sortedOps.push_back(op);
    for (auto succ : adjList[op]) {
      if (--indegree[succ] == 0) {
        que.push(succ);
      }
    }
  }
  if (sortedOps.size() != indegree.size()) {
    LOGE << "soyred op size != indegree size, there may be a cycle in the "
            "graph\n";
    return LogicalResult::failure();
  }
  for (auto op : sortedOps) {
    if (terminator) {
      op->moveBefore(terminator);
    } else {
       op->moveBefore(&block, block.end()); 
    }
  }
  return LogicalResult::success();
}
// 递归 DFS 访问
void dfsVisitWithPreds(mlir::Operation *op,
                       llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 4>> &predMap,
                       llvm::SmallPtrSetImpl<mlir::Operation *> &visited,
                       llvm::SmallVectorImpl<mlir::Operation *> &sortedOps) {
    if (!visited.insert(op).second) {
        return;
    }

    // 优先遍历所有前驱依赖（定义当前Op所需的那些Op）
    for (mlir::Operation *pred : predMap[op]) {
        dfsVisitWithPreds(pred, predMap, visited, sortedOps);
    }

    // 前驱都处理完了，再放入自己，天然形成正向拓扑序
    sortedOps.push_back(op);
}

LogicalResult reorderOpsRPO(mlir::Block &block, mlir::MLIRContext *ctx) {
  if (block.empty())
    return LogicalResult::success();
  mlir::Operation *terminator =
      block.mightHaveTerminator() ? block.getTerminator() : nullptr;
  // 前驱依赖图：Op -> 它依赖的前置 Op 集合
  llvm::DenseMap<mlir::Operation *, llvm::SetVector<mlir::Operation *>> predMap;
  mlir::DenseMap<mlir::Operation *, int64_t> depth_map;
  // 1. 利用 getUsers() 高效反推构建前驱图
  for (mlir::Operation &op : block) {
    if (&op == terminator)
      continue;
    for (mlir::Value result : op.getResults()) {
      for (mlir::Operation *user : result.getUsers()) {
        mlir::Operation *userAncestor = getAncestorInBlock(user, &block);
        if (userAncestor && userAncestor != &op && userAncestor != terminator) {
          predMap[userAncestor].insert(&op);
        }
      }
    }
    getMaxPath(&op, depth_map);
  }
  // 2.根据路径长度排序，路径短的优先
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *, 4>> sortedPredMap;
for (auto &item : predMap) {
    sortedPredMap[item.first] = item.second.takeVector(); // 取出底层 vector
    std::sort(sortedPredMap[item.first].begin(), sortedPredMap[item.first].end(),
              [&depth_map](mlir::Operation *a, mlir::Operation *b) {
                  return depth_map[a] < depth_map[b];
              });
}

  //3. rpo排序
    llvm::SmallPtrSet<mlir::Operation *, 32> visited;
    llvm::SmallVector<mlir::Operation *> sortedOps;

    // 2. DFS 遍历收集 RPO 序列
    for (mlir::Operation &op : block) {
        if (&op == terminator) continue;
        dfsVisitWithPreds(&op, sortedPredMap, visited, sortedOps);
    }

    // 3. 物理重排
    for (mlir::Operation *op : sortedOps) {
        if (terminator) op->moveBefore(terminator);
        else  op->moveBefore(&block, block.end()); 
    }

  return LogicalResult::success();
}
class OpreorderPass : public OpReOrderPassBase<OpreorderPass> {
public:
  OpreorderPass() {}
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto algo_enum = tbc::utils::symbolizeReorderAlgo(algo);
    if (!algo_enum.has_value()) {
      funcOp->emitError("No reorder algorithm named:" + algo);
      signalPassFailure();
      return;
    }
    auto &block = funcOp.getBody().front();
    auto ctx = funcOp->getContext();
    if (algo_enum.value() == tbc::utils::ReorderAlgo::RPO) {
      if (failed(reorderOpsRPO(block, ctx))) {
        funcOp->emitError("Failed to reorder ops with RPO algorithm");
        signalPassFailure();
        return;
      }
    } else if (algo_enum.value() == tbc::utils::ReorderAlgo::KAHN) {
      if (failed(reorderOpsKahn(block, ctx))) {
        funcOp->emitError("Failed to reorder ops with KAHN algorithm");
        signalPassFailure();
        return;
      }
    } else {
      funcOp->emitError("Unsupported reorder algorithm:" + algo);
      signalPassFailure();
      return;
    }
  }
};

std::unique_ptr<OperationPass<mlir::func::FuncOp>> createOpreorderPass() {
  return std::make_unique<OpreorderPass>();
}
}; // namespace tbc::kls
