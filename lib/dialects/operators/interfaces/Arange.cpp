//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"


// ArangeOp is special, will convert to WeightOp
void ops::ArangeOp::shape_inference() {
  float start = 0;
  float step = 1;
  assert(module::isWeight(getEnd()));
  auto end_op = getEnd().getDefiningOp<ops::WeightOp>();
  auto end_data = end_op.read<float>();
  auto end = end_data->at(0);
  if (module::isNone(getStart()) == false) {
    assert(module::isWeight(getStart()));
    auto start_op = getStart().getDefiningOp<ops::WeightOp>();
    auto start_data = start_op.read<float>();
    start = start_data->at(0);
  }
  if (module::isNone(getStep()) == false) {
    assert(module::isWeight(getStep()));
    auto step_op = getStep().getDefiningOp<ops::WeightOp>();
    auto step_data = step_op.read<float>();
    step = step_data->at(0);
    assert(step != 0);
  }
  std::vector<float> data;
  for (int i = start; i < end; i += step) {
    data.push_back(i);
  }
  auto op = getOperation();
  OpBuilder builder(module::getCtx());
  builder.setInsertionPointAfter(op);
  auto weight_type =
      RankedTensorType::get({(int64_t)data.size()}, builder.getF32Type());
  auto new_op = ops::WeightOp::create(op, "arange", data, weight_type);
  getOutput().replaceAllUsesWith(new_op);
}
void ops::ArangeOp::type_inference() { module::setElementType(getOutput(), mlir::IntegerType::get(this->getContext(), 64)); }
