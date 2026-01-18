#include "dialects/hals/IR/hals.h"
#include "dialects/hals/transforms/pass.h"
#include "support/module.h"
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
      auto groupOp=hals::PackedWeightGroupOp::Merge(packed_weight_groups);
      //assgin tensor addr
      auto offsets=groupOp.getOffsets();
      //return values
      for(auto &p:offsets){
        auto type=mlir::cast<HalTensorType>(p.first.getType());
        auto params=type.parse_params();
        params.addr=p.second;
        p.first.setType(HalTensorType::get(module::getCtx(),params));
      }
      // weight values
      auto valu_map=groupOp.getReturnValueMap();
      for(auto &p:valu_map){
        p.second.setType(p.first.getType());
      }
    }
  };
  std::unique_ptr<mlir::OperationPass<FuncOp>> createFusePackedWeightGroupToOnePass(){
    return std::make_unique<FusePackedWeightGroupToOnePass>();
  }
}
