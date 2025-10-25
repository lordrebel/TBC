#include "dialects/hals/IR/hals.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace tbc::hals{
  int64_t HalTensorType::getMemorySize() const {
    llvm_unreachable("not implement");
  }
  
  cascade_param_t CascadeAttr::parse_params() const{
    cascade_param_t params;
    params.block_size = llvm::SmallVector<int64_t, 4>(getBlockParams().begin(), getBlockParams().end());
    params.buffer_addrs = llvm::SmallVector<int64_t, 4>(getBufferAddrs().begin(), getBufferAddrs().end());
    params.is_cascade = getIsCascade();
    return params;

  }

  hal_tensor_params_t HalTensorType::parse_params() const{
    hal_tensor_params_t params;
    auto ranked_type = GetRankedTensorType();
    params.shape = llvm::SmallVector<int64_t, 4>(ranked_type.getShape().begin(), ranked_type.getShape().end());
    params.element_type = ranked_type.getElementType();
    params.memory_space = getMemorySpace();
    params.layout = getLayout();
    params.addr = getAddr();
    params.kind = getKind();
    if (auto cascadeAttr = getCascade()) {
      params.cascade = cascadeAttr.parse_params();
   }
    return params;
  }
}
