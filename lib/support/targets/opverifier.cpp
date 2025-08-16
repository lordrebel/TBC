#include "support/module.h"
#include "support/targets/target.h"

namespace tbc::tgt {
class NPUV1OpVerifier : public ITargetOpVerifier<NPUV1OpVerifier> {
public:
  template <typename OpType>
  llvm::LogicalResult verifyKernelOpImpl(OpType op) {
    // Default implementation can be empty or provide basic checks
    return llvm::LogicalResult::success();
  }

  template <typename OpType>
  llvm::LogicalResult verifyHalOpImpl(OpType op) {
    // Default implementation can be empty or provide basic checks
    return llvm::LogicalResult::success();
  }
};
class NPUV2OpVerifier : public ITargetOpVerifier<NPUV2OpVerifier> {
public:
  template <typename OpType>
  llvm::LogicalResult verifyKernelOpImpl(OpType op) {
    // Default implementation can be empty or provide basic checks
    return llvm::LogicalResult::success();
  }

  template <typename OpType>
  llvm::LogicalResult verifyHalOpImpl(OpType op) {
    // Default implementation can be empty or provide basic checks
    return llvm::LogicalResult::success();
  }
};

NPUV1OpVerifier &getNPUV1Verifier() {
  static NPUV1OpVerifier instance; // 懒加载，线程安全
  return instance;
}

NPUV2OpVerifier &getNPUV2Verifier() {
  static NPUV2OpVerifier instance;
  return instance;
}

// real implementation of the target op verifier
#include "op_verify_inc/npuv1.cpp.inc"
#include "op_verify_inc/npuv2.cpp.inc"

template <>
llvm::LogicalResult verifyHalOp<tbc::hals::EltwiseOp>(tbc::hals::EltwiseOp op) {
  switch (tbc::module::getTarget()) {
  case tbc::utils::Target::NPU_V1:
    return getNPUV1Verifier().template verifyHalOp<tbc::hals::EltwiseOp>(op);
  case tbc::utils::Target::NPU_V2:
    return getNPUV2Verifier().template verifyHalOp<tbc::hals::EltwiseOp>(op);
  default:
    LOGE << ""
         << "Unsupported target for EltwiseOp verification: "
         << tbc::module::getTarget();
    return llvm::failure();
  }
}

} // namespace tbc::tgt
