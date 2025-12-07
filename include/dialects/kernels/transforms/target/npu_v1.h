#pragma once
#include<mlir/Pass/PassManager.h>
#include "dialects/kernels/transforms/target/targetRegistry.h"
namespace tbc::kls {

  void RegisterNpuV1TargetPasses(mlir::PassManager &pm);
}
