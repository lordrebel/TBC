//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

void ops::GatherElementsOp::shape_inference() {
  auto indices_shape = module::getShape(getIndices());
  module::setShapeOrVerify(getOutput(), indices_shape);
}
