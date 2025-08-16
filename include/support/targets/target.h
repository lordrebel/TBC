#pragma once

#include "mlir/IR/Operation.h"
#include "llvm/Support/LogicalResult.h"

#include "dialects/kernels/IR/Kernels.h"
#include "dialects/hals/IR/hals.h"
namespace tbc {
namespace tgt{
  class ITraget{

  public:
    virtual ~ITraget() = default;
  };

  template <typename DerivedTarget>
  class ITargetOpVerifier {
    public:
    template <typename OpType>
      llvm::LogicalResult verifyKernelOp(OpType op) {
        static_cast<DerivedTarget *>(this)-> template verifyKernelOpImpl<OpType>(op);
      }
      template <typename OpType>
      llvm::LogicalResult verifyHalOp(OpType op) {
        static_cast<DerivedTarget *>(this)-> template verifyHalOpImpl<OpType>(op);
      }
  };

//-----------------------------------------------------------------
// Helper for tareget op verification
//-----------------------------------------------------------------
template <typename OpTy>
llvm::LogicalResult verifyHalOp(OpTy op){
  return llvm::success();
}
template <typename OpTy>
llvm::LogicalResult verifyKernelOp(OpTy op){
  return llvm::success();
}

template<> llvm::LogicalResult verifyHalOp<tbc::hals::EltwiseOp>(tbc::hals::EltwiseOp op);

//-----------------------------------------------------------------
// Target class
//-----------------------------------------------------------------
}

}
