#include "dialects/kernels/transforms/pass.h"
#include"support/module.h"
#include <llvm/Support/Debug.h>
#define DEBUG_TYPE "assign_target_pass"

using namespace llvm;
using namespace mlir;
namespace tbc {
namespace kls {
  class AssginTargetPass : public AssginTargetBase<AssginTargetPass> {
public:
  AssginTargetPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    auto target_enum = tbc::utils::symbolizeTarget(target);
    if(!target_enum.has_value()) {
      mOp->emitError("No target named:"+target);
      signalPassFailure();
      return;
    }

    module::setTarget(target_enum.value());

  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAssginTargetPass() {
  return std::make_unique<AssginTargetPass>();
}

}
}
