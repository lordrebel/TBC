#pragma once
#include<mlir/Pass/PassManager.h>
namespace tbc::kls {

  void RegisterNpuV2TargetPasses(mlir::PassManager &pm);
}
