//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"


void ops::SqueezeOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto in_dims = in_shape.size();
  auto axes = module::getI64Array(getAxesAttr());
  std::vector<int64_t> axes_ = *axes;
  for (auto &a : axes_) {
    if (a < 0) {
      a += in_dims;
    }
  }
  std::vector<int64_t> out_shape;
  for (int i = 0; i < in_dims; ++i) {
    if (axes_.empty()) {
      if (in_shape[i] != 1) {
        out_shape.push_back(in_shape[i]);
      }
    } else {
      if ((std::find(axes_.begin(), axes_.end(), i) == axes_.end()) || (in_shape[i] != 1)) {
        out_shape.push_back(in_shape[i]);
      } else {
        assert(in_shape[i]);
      }
    }
  }
  if (out_shape.empty()) {
    out_shape.push_back(1);
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}
