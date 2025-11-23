#pragma once
#include"dialects/hals/IR/HalEnum.h.inc"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
namespace tbc::hals{
struct cascade_param_t {
    llvm::SmallVector<int64_t, 4> block_size{-1,-1,-1,-1};
    llvm::SmallVector<int64_t, 4> buffer_addrs;
    bool is_cascade=false;
} ;

struct hal_tensor_params_t {
  cascade_param_t cascade;
  int64_t addr;
  llvm::SmallVector<int64_t, 4> shape{-1,-1,-1,-1};
  mlir::Type element_type;
  MemorySpace memory_space;
  StorageLayout layout;
  TensorKind kind;

} ;
}
