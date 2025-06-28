#include "dialects/operators/transforms/pass.h"
#include "interfaces/typeInfer_interface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "support/module.h"
#include <llvm/Support/Debug.h>
#define DEBUG_TYPE "type_infer"
using namespace llvm;
using namespace mlir;
namespace tbc {
namespace ops {

class TypeInferPass : public TypeInferBase<TypeInferPass> {
public:
  TypeInferPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();

    // Do shape infer
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](TypeInferInterface op) {
        LLVM_DEBUG(llvm::dbgs() << "type infer: " << op << "\n";);
        op.type_inference();
      });
    }
    module::updateModuleTypes();
  }
};

std::unique_ptr<mlir::OperationPass<ModuleOp>> createTypeInferPass() {
  return std::make_unique<TypeInferPass>();
}
} // namespace ops
} // namespace tbc
