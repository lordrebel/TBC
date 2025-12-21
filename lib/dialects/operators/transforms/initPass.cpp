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
using namespace llvm;

using namespace mlir;
namespace tbc {
namespace ops {

class InitPass : public InitBase<InitPass> {
public:
  InitPass() {}
  void runOnOperation() override {
    tbc::support::initLogger();
    auto mOp = getOperation();
    module::init(mOp);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createInitPass() {
  return std::make_unique<InitPass>();
}
} // namespace ops
} // namespace tbc
