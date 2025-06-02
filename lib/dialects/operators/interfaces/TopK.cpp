//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"


void ops::TopKOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  int64_t K = -1;
  if (module::isShape(getKT())) {
    auto kt_vec = module::getShapeTensorValue(getKT());
    assert(kt_vec.size() == 1);
    K = kt_vec[0];
    setK(K);
  } else {
    K = getK();
  }
  int64_t axis = getAxis();
  int64_t rank = input_shape.size();
  axis = axis < 0 ? axis + rank : axis;
  setAxis(axis);
  std::vector<int64_t> output_shape(input_shape.size());
  for (int i =0; i < input_shape.size(); i++) {
    if (i == axis) {
      output_shape[i] = K;
    } else {
      output_shape[i] = input_shape[i];
    }
  }
  module::setShapeOrVerify(getValues(), output_shape);
  module::setShapeOrVerify(getIndices(), output_shape);
}


