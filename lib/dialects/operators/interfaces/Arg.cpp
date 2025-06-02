//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"



void ops::ArgOp::shape_inference() {
  int64_t axis = getAxis();
  auto input_shape = module::getShape(getInput());
  const int input_dims = input_shape.size();
  if (axis < 0) {
      axis += input_dims;
      setAxis(axis);
  }
  std::vector<int64_t> output_shape;
  output_shape.reserve(input_dims);
  output_shape.assign(input_shape.begin(), input_shape.begin() + axis);
  if (getKeepdims()) {
    output_shape.push_back(1);
  }
  output_shape.insert(output_shape.end(), input_shape.begin() + axis + 1, input_shape.end());
  module::setShapeOrVerify(getIndices(), output_shape);
  if (!module::isNone(getValues())) {
    module::setShapeOrVerify(getValues(), output_shape);
  }
}
