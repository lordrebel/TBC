#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"
#include<vector>
namespace tbc::hals{

  struct LgInfo{
      // in tensors
  llvm::SmallSetVector<mlir::Value, 16>group_ins;
  // out tensors
  llvm::SmallSetVector<mlir::Value, 16> group_outs;
  //weights in current layergroup
  llvm::SmallSetVector<mlir::Value, 16> group_weights;

  //ops in currrent layergroup
    llvm::SetVector<mlir::Operation *> group_ops; 

  };

  struct LayerGroupIR{
    llvm::SetVector<mlir::Operation *> layerOps;
    llvm::SetVector<mlir::Value> values;
    std::vector<LgInfo> lg_infos;
    

  };

}
