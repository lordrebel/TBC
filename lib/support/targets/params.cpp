#include "support/targets/params.h"
#include "support/module.h"

namespace tbc::tgt{
  DATA_TYPE_T getDataType(mlir::Value v) {
  auto type = module::getStorageType(v);
  return getDataType(type);
}
DATA_TYPE_T getDataType(mlir::Type type) {
  auto bits = type.getIntOrFloatBitWidth();
  if (type.isUnsignedInteger()) {
    switch (bits) {
    case 4:
      return DTYPE_UINT4;
    case 8:
      return DTYPE_UINT8;
    case 16:
      return DTYPE_UINT16;
    case 32:
      return DTYPE_UINT32;
    default:
      break;
    }
  } else if (type.isSignedInteger() || type.isSignlessInteger()) {
    switch (bits) {
    case 4:
      return DTYPE_INT4;
    case 8:
      return DTYPE_INT8;
    case 16:
      return DTYPE_INT16;
    case 32:
      return DTYPE_INT32;
    default:
      break;
    }
  } else if (type.isF32()) {
    return DTYPE_FP32;
  } else if (type.isBF16()) {
    return DTYPE_BFP16;
  } else if (type.isF16()) {
    return DTYPE_FP16;
  } else if (type.isFloat8E4M3FN()) {
    return DTYPE_F8E4M3;
  } else if (type.isFloat8E5M2()) {
    return DTYPE_F8E5M2;
  }
  type.dump();
  llvm_unreachable("Unsupport type \n");
  return DTYPE_FP32;
}
}
