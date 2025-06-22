//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "dialects/operators/IR/operator.h"
#include "support/module.h"

void ops::SwapDimInnerOp::shape_inference() {
  common_shape_inference(getOperation());
}

void ops::SwapDimInnerOp::type_inference() {
  auto input=getInput();
  auto output=getOutput();
  module::setElementType(output, module::getElementType(input));
}
