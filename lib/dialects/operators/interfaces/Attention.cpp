//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"
void ops::AttentionOp::shape_inference() {}
void ops::AttentionOp::type_inference() {
  auto type=module::getElementType(getInput());
  module::setElementType(getOutput(), type);
}
