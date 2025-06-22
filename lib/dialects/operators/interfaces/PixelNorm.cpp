//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"


void ops::PixelNormOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  std::vector<int64_t> wb_shape(input_shape.size(), 1);
  wb_shape[1] = input_shape[1];
  RankedTensorType newType;
  if (auto weight_op = dyn_cast_or_null<WeightOp>(getWeight().getDefiningOp())) {
    newType = RankedTensorType::get(wb_shape, module::getElementType(weight_op));
    getWeight().setType(newType);
  }
  if (auto bias_op = dyn_cast_or_null<WeightOp>(getBias().getDefiningOp())) {
    newType = RankedTensorType::get(wb_shape, module::getElementType(bias_op));
    getBias().setType(newType);
  }
  common_shape_inference(getOperation());
}
void ops::PixelNormOp::type_inference() {
  auto input = getInput();
  auto output = getOutput();
  module::setElementType(output, module::getElementType(input));
}
