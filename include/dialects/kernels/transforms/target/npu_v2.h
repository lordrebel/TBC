#pragma once
#include<mlir/Pass/PassManager.h>
#include "dialects/kernels/transforms/target/targetRegistry.h"
namespace tbc::kls {

  void RegisterNpuV2TargetPasses(mlir::PassManager &pm);
}
