//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

void ops::NmsOp::shape_inference() {
  int class_num = module::getShape(getInputs()[1])[1];
  int max_output_size_per_class = 0;
  if (module::isShape(getInputs()[2])) {
    auto vec = module::getShapeTensorValue(getInputs()[2]);
    assert(vec.size() == 1);
    max_output_size_per_class = vec[0];
  } else {
    max_output_size_per_class = getMaxOutputSize();
  }
  std::vector<int64_t> output_shape{0,3};
  output_shape[0] = class_num * max_output_size_per_class;
  module::setShapeOrVerify(getOutput(), output_shape);
}
