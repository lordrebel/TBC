//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"
#include <cstdint>

// SizeOp is special, will convert to WeightOp
void ops::SizeOp::shape_inference() {
  auto shape = module::getShape(getInput());
  std::vector<int64_t> data;
  if (getAxis().has_value()) {
    auto axis = getAxis().value();
    if (axis < 0) {
      axis += shape.size();
    }
    data.push_back(shape[axis]);
  } else {
    for (auto s : shape) {
      data.push_back(s);
    }
  }
  auto op = getOperation();
  OpBuilder builder(module::getCtx());
  builder.setInsertionPointAfter(op);
  auto weight_type =
      RankedTensorType::get({(int64_t)data.size()}, builder.getIntegerType(64));
  auto new_op = ops::WeightOp::create(op, "size", data, weight_type);
  getOutput().replaceAllUsesWith(new_op);
}
void ops::SizeOp::type_inference() {
  auto output = getOutput();
  module::setElementType(output, mlir::IntegerType::get(
      module::getCtx(), 64)); // Size is always int64_t, so use IntegerType with 64 bits
}
