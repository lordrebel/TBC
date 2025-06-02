//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"


void ops::MeshGridOp::shape_inference() {
  int64_t input_num = getInputs().size();
  int64_t length = 1;
  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < input_num; ++i) {
    int64_t idx = getIsReverse() ? (input_num - 1 - i) : i;
    auto shape = module::getShape(getInputs()[idx]);
    out_shape.push_back(shape[0]);
    length *= shape[0];
  }
  for (int i = 0; i < input_num; ++i) {
    auto out = getResult(i);
    module::setShapeOrVerify(out, out_shape);
  }
}
