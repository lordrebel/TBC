#include "conversions/conversion.h"
#include "conversions/OperatorsToKernels/opertorsTokernels.h"
#include "mlir/Pass/Pass.h"
namespace tbc{
  std::unique_ptr<mlir::Pass> createConvertOperatorsToKernels(){
    return std::make_unique<ConvertOperatorsToKernels>();
  }

  void ConvertOperatorsToKernels::runOnOperation(){
    llvm_unreachable("not implemented yet");
  }
}
