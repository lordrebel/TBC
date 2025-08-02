#include "dialects/hals/IR/hals.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace tbc::hals{
  int64_t HalTensorType::getMemorySize() const {
    llvm_unreachable("not implement");
  }
  
}
