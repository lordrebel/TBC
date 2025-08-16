#pragma once
#include "mlir/IR/Value.h"


namespace tbc::tgt {
  typedef enum {
  DTYPE_FP32 = 0,
  DTYPE_FP16 = 1,
  DTYPE_INT8 = 2,
  DTYPE_UINT8 = 3,
  DTYPE_INT16 = 4,
  DTYPE_UINT16 = 5,
  DTYPE_INT32 = 6,
  DTYPE_UINT32 = 7,
  DTYPE_BFP16 = 8,
  DTYPE_INT4 = 9,
  DTYPE_UINT4 = 10,
  DTYPE_FP20 = 11,
  DTYPE_F8E5M2 = 12,
  DTYPE_F8E4M3 = 13,
  DTYPE_UNKNOWN = -1,
} DATA_TYPE_T;

DATA_TYPE_T getDataType(mlir::Value v); 
DATA_TYPE_T getDataType(mlir::Type type);
}
