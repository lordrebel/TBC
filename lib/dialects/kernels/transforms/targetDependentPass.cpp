#include "dialects/kernels/transforms/pass.h"
#include "dialects/kernels/transforms/target/npu_v1.h"
#include "dialects/kernels/transforms/target/npu_v2.h"
#include "dialects/kernels/transforms/target/targetRegistry.h"
namespace tbc::kls {
class TargetDependentPass: public TargetDependentPassBase<TargetDependentPass> {
  public:
  TargetDependentPass() {}
  void runOnOperation() override {
    TargetDependentPassRegistry::Regist(utils::Target::NPU_V1, RegisterNpuV1TargetPasses);
    TargetDependentPassRegistry::Regist(utils::Target::NPU_V2, RegisterNpuV2TargetPasses);

    auto mOp = getOperation();
    mlir::PassManager pm(mOp->getContext());
    TargetDependentPassRegistry::Get(module::getTarget(), pm);
    if (failed(pm.run(mOp))) {
      signalPassFailure();
    }

  }
};

std::unique_ptr<mlir::OperationPass<ModuleOp>> createTargetDependentPass(){
  return std::make_unique<TargetDependentPass>();
}
}
