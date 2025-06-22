//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "dialects/operators/IR/operator.h"
#include "support/module.h"


void ops::UpsampleOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  int dim = input_shape.size();
  assert(dim >= 2);
  std::vector<int64_t> out_shape(dim);
  int64_t scale_h = this->getScaleH();
  int64_t scale_w = this->getScaleW();
  for (int i = 0; i < dim; i++) {
    if (i == dim - 2) {
      out_shape[i] = input_shape[i] * scale_h;
    } else if (i == dim - 1) {
      out_shape[i] = input_shape[i] * scale_w;
    } else {
      out_shape[i] = input_shape[i];
    }
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}

void ops::UpsampleOp::type_inference() {
  auto input=getInput();
  auto output=getOutput();
  module::setElementType(output, module::getElementType(input));
}
