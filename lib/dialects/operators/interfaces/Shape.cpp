//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"
#include "llvm/Support/ErrorHandling.h"


void ops::ShapeOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  bool no_slice = true;
  int64_t input_dims = input_shape.size();
  int64_t start = getStart().has_value() ? getStart().value() : 0;
  int64_t end = getEnd().has_value() ? getEnd().value() : input_dims;
  end = std::clamp(end, 0L, input_dims);
  if (getStart().has_value()) {
    removeStartAttr();
  }
  if (getEnd().has_value()) {
    removeEndAttr();
  }
  if (start != 0 || end != input_dims) {
    no_slice = false;
  }
  std::vector<int64_t> output_shape({(int64_t)input_shape.size()});
  if (!no_slice) {
    llvm_unreachable("not support slice shape op fpr shape infer");
  } else {
    module::setShapeOrVerify(getOutput(), output_shape);
    module::bindShapeTensorValue(getOutput(), module::getShape(getInput()));
  }
}
