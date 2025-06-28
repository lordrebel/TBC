#include "dialects/operators/transforms/platform/platormPassRegistry.h"

namespace tbc::ops {
class PrintOpnamePass : public PrintOpnameBase<PrintOpnamePass> {
public:
  PrintOpnamePass() {}
  void runOnOperation() override {
    auto mOp = getOperation();

    // Do shape infer
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](Operation * op) {
        llvm::outs() << "op: " << op->getName() << "\n";
      });
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createPrintOpNamePass() {
  return std::make_unique<PrintOpnamePass>();
}
} // namespace tbc::ops
