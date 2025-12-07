#include"dialects/kernels/transforms/target/npu_v1.h"
#include"dialects/kernels/transforms/target/npu_v2.h"

namespace tbc::kls {
  //implement target pass here


  void RegisterNpuV1TargetPasses(mlir::PassManager & pm) {
    // Add NPU V1 specific passes here
    pm.addPass(createExtraOptimizeNpuV1Pass());
  }
  void RegisterNpuV2TargetPasses(mlir::PassManager & pm) {
    // Add NPU V2 specific passes here
  }

}
