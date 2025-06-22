//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"


static inline double hswish(double x) {
  return x * std::max(0.0, std::min(1.0, x / 6 + 0.5));
}

void ops::HardSwishOp::shape_inference() {
  common_shape_inference(getOperation());
}
void ops::HardSwishOp::type_inference() {
  common_type_inference(getOperation());
}
