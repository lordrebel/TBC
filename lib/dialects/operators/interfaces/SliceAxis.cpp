//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"


void ops::SliceAxisOp::shape_inference() {
  float start = 0;
  float step = 1;
  float end;
  auto in_shape = module::getShape(getInput());
  assert(module::isWeight(getAxis()));
  auto axis_op = getAxis().getDefiningOp<ops::WeightOp>();
  auto axis_data = axis_op.read<float>();
  auto axis = axis_data->at(0);
  if (module::isNone(getEnd()) == false) {
    if (module::isWeight(getEnd())) {
      auto end_op = getEnd().getDefiningOp<ops::WeightOp>();
      auto end_data = end_op.read<float>();
      end = end_data->at(0);
    } else {
      end = in_shape[(int)axis];
    }
  }
  if (module::isNone(getStart()) == false) {
    if (module::isWeight(getStart())) {
      auto start_op = getStart().getDefiningOp<ops::WeightOp>();
      auto start_data = start_op.read<float>();
      start = start_data->at(0);
    } else {
      start = 0;
    }
  }
  if (module::isNone(getStep()) == false) {
    assert(module::isWeight(getStep()));
    auto step_op = getStep().getDefiningOp<ops::WeightOp>();
    auto step_data = step_op.read<float>();
    step = step_data->at(0);
    assert(step != 0);
  }
  auto dims = in_shape.size();
  if (axis < 0) {
    axis += dims;
  }
  if (start < 0) {
    if(end - start == 1 && step == 1)
      end = start + in_shape[axis] + 1;
    start += in_shape[axis];
  }
  if (end < 0) {
    end += in_shape[axis];
  } else if (end > in_shape[axis]) {
    end = in_shape[axis];
  }
  std::vector<int64_t> out_shape(in_shape);
  out_shape[axis] = (end - start + step - 1) / step;
  module::setShapeOrVerify(getOutput(), out_shape);
}

void ops::SliceAxisOp::type_inference() {
  auto input=getInput();
  auto output=getOutput();
  module::setElementType(output, module::getElementType(input));
}
