//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

void ops::ReverseOp::shape_inference() {
  common_shape_inference(getOperation());
}
void ops::ReverseOp::type_inference() {
  common_type_inference(getOperation());
}

