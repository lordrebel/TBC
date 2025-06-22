//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "dialects/operators/IR/operator.h"
#include "mlir/IR/Types.h"
#include "support/module.h"

void ops::QuantizeLinearOp::shape_inference() {
  common_shape_inference(getOperation());
}
void ops::QuantizeLinearOp::type_inference() {
  DataType dtype= tbc::utils::symbolizeDataType(getDtypeAttr().getValue()).value();
  Type type =module::DatatypeEnumToType(dtype,getContext());
  module::setElementType(getOutput(), type);
}
