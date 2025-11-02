#pragma once
#include<mlir/Pass/PassManager.h>
namespace tbc::kls {

  void RegisterNpuV1TargetPasses(mlir::PassManager &pm);
}
