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

void ops::ConcatOp::shape_inference() {
  auto axis_ = getAxis();
  auto in0_shape = module::getShape(getInputs()[0]);
  if (axis_ < 0) {
    axis_ += in0_shape.size();
    setAxis(axis_);
  }
  int64_t shape_axis = 0;
  for (auto inp : getInputs()) {
    auto shape = module::getShape(inp);
    shape_axis += shape[axis_];
  }
  std::vector<int64_t> out_shape(in0_shape);
  out_shape[axis_] = shape_axis;
  module::setShapeOrVerify(getOutput(), out_shape);
  if (llvm::find_if(getOperands(), module::isShape) != getOperands().end()) {
    llvm_unreachable("not support shape inference for concat op yet. ");
  }
}
