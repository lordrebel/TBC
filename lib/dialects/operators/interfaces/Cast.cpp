//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"
#include "support/utils.h"
#include "llvm/Support/ErrorHandling.h"


void ops::CastOp::shape_inference() { common_shape_inference(getOperation()); }
void ops::CastOp::type_inference() {
  DataType dtype= tbc::utils::symbolizeDataType(getDtypeAttr().getValue()).value();
  Type type =module::DatatypeEnumToType(dtype,getContext());
  module::setElementType(getOutput(), type);
}
