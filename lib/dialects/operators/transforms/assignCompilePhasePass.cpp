#include "dialects/operators/transforms/pass.h"
#include"support/module.h"
#include <llvm/Support/Debug.h>
#define DEBUG_TYPE "assign_compile_phase"

using namespace llvm;
using namespace mlir;
namespace tbc {
namespace ops {
  class AssginCompilePhasePass : public AssginCompilePhaseBase<AssginCompilePhasePass> {
public:
  AssginCompilePhasePass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    auto compile_phase_enum = tbc::utils::symbolizeCompilePhase(compile_phase);
    if(!compile_phase_enum.has_value()) {
      mOp->emitError("No compile phase named"+compile_phase);
      signalPassFailure();
      return;
    }

    module::setCompilePhase(compile_phase_enum.value());

  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAssignCompilePhasePass() {
  return std::make_unique<AssginCompilePhasePass>();
}

}
}
