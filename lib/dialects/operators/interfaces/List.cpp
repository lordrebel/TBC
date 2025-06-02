//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

// ListOp is special, will convert to WeightOp
void ops::ListOp::shape_inference() {
  int64_t num_outputs = 0;
  for (auto in : getInputs()) {
    if (module::isNone(in)) {
      continue;
    }
    num_outputs += module::getNumElements(in);
  }
  std::vector<int64_t> new_shape = {num_outputs};
  module::setShapeOrVerify(getOutput(), new_shape);
}
