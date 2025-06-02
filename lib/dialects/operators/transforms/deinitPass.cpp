//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "dialects/operators/transforms/pass.h"
#include"support/module.h"
#include "support/utils.h"
using namespace llvm;


namespace tbc {
namespace ops {

class DeinitPass : public DeinitBase<DeinitPass> {
public:
  DeinitPass() {}
  void runOnOperation() override {
    auto compile_phase = module::getCompilePhase();
    if (compile_phase >= utils::CompilePhase::IMPORTED) {
      return;
    }
    module::removeUnusedOp();
    module::saveWeight();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createDeinitPass() {
  return std::make_unique<DeinitPass>();
}
} // namespace ops
} // namespace tbc
