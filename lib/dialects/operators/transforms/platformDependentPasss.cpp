#include "dialects/operators/transforms/pass.h"
#include"support/module.h"
#include"dialects/operators/transforms/platform/platormPassRegistry.h"
#include"mlir/Pass/PassManager.h"
#include "support/utils.h"
using namespace llvm;

using namespace mlir;
namespace tbc {
namespace ops {

class PlatformDependentPass : public PlatformDependentBase<PlatformDependentPass> {
public:
  PlatformDependentPass() {}
  void runOnOperation() override {
    PlatformPassRegistry::Initialize();
    auto mOp = getOperation();
    mlir::PassManager pm(mOp->getContext());
    Platform platform = tbc::module::getPlatform();
    PlatformPassRegistry::Get(platform, pm);

    if (failed(pm.run(mOp))) {
      signalPassFailure();
    }

  }
};

std::unique_ptr<OperationPass<ModuleOp>> createPlatformDependentPass() {
  return std::make_unique<PlatformDependentPass>();
}
} // namespace ops
} // namespace tbc
