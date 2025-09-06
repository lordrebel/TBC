#include "conversions/conversion.h"
#include "mlir/Pass/Pass.h"
#include "conversions/KernelsToHals/kernelsTohals.h"

namespace tbc{

  std::unique_ptr<mlir::Pass> createConvertKernelsToHals(){
    return std::make_unique<ConvertKernelsToHals>();
  }

  void ConvertKernelsToHals::runOnOperation(){
    llvm_unreachable("not implemented yet");
  }

}
