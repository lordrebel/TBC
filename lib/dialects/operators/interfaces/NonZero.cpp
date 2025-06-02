//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

void ops::NonZeroOp::shape_inference() {
  const int order = getOrder().str() == "ColMajor" ? 0 : 1;
  int64_t num_elem = module::getNumElements(getInput());
  const auto shape = module::getShape(getInput());
  int64_t dims = shape.size();
  std::vector<int64_t> output_shape;
  if (order) {
    output_shape.push_back(dims);
    output_shape.push_back(num_elem);
  } else {
    output_shape.push_back(num_elem);
    output_shape.push_back(dims);
  }
  module::setShapeOrVerify(getOutput(), output_shape);
}
