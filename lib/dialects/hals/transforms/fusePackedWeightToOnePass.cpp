#include "dialects/hals/IR/hals.h"
#include "dialects/hals/transforms/pass.h"
#include <memory>
namespace tbc::hals {

  class FusePackedWeightGroupToOnePass: public FusePackedWeightGroupToOnePassBase<FusePackedWeightGroupToOnePass>{
    public:
    FusePackedWeightGroupToOnePass() {}
    void runOnOperation() override {
      std::vector<hals::PackedWeightGroupOp> packed_weight_groups;
      getOperation().walk([&](hals::PackedWeightGroupOp op) {
        packed_weight_groups.push_back(op);
      });
      hals::PackedWeightGroupOp::Merge(packed_weight_groups);
      //the merged packed_weight_group op from the function.
    
    }
  };
  std::unique_ptr<mlir::OperationPass<FuncOp>> createFusePackedWeightGroupToOnePass(){
    return std::make_unique<FusePackedWeightGroupToOnePass>();
  }
}
