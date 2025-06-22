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
#include "llvm/Support/ErrorHandling.h"



void ops::WhereOp::shape_inference() {
  broadcast_shape_inference(getOperation());
  // support case input/output both shape.
  // cond/x/y all weight/shape, and weight is integer
  std::vector<std::vector<int64_t>> input_shapes_v;
  if (module::isShape(getCond())) {
    auto input_shape_v = module::getShapeTensorValue(getCond());
    input_shapes_v.push_back(input_shape_v);
  } else if (module::isWeight(getCond())) {
    auto data = getCond().getDefiningOp<ops::WeightOp>().read_as_float();
    std::vector<int64_t> data_v(data->begin(), data->end());
    input_shapes_v.push_back(data_v);
  }
  if (getXIsConst()) {
    auto x_const_v = getXConstVal().convertToDouble();
    if (x_const_v == floor(x_const_v)) {
      input_shapes_v.push_back({static_cast<int>(x_const_v)});
    }
  } else if (module::isShape(getTbrn())) {
    auto input_shape_v = module::getShapeTensorValue(getTbrn());
    input_shapes_v.push_back(input_shape_v);
  } else if (module::isWeight(getTbrn())) {
    auto data = getTbrn().getDefiningOp<ops::WeightOp>().read_as_float();
    if (std::all_of(data->begin(), data->end(),
                    [](auto &x) { return x == floor(x); })) {
      std::vector<int64_t> data_v(data->begin(), data->end());
      input_shapes_v.push_back(data_v);
    }
  }
  if (getYIsConst()) {
    auto x_const_v = getYConstVal().convertToDouble();
    if (x_const_v == floor(x_const_v)) {
      input_shapes_v.push_back({static_cast<int>(x_const_v)});
    }
  } else if (module::isShape(getFbrn())) {
    auto input_shape_v = module::getShapeTensorValue(getFbrn());
    input_shapes_v.push_back(input_shape_v);
  } else if (module::isWeight(getFbrn())) {
    auto data = getFbrn().getDefiningOp<ops::WeightOp>().read_as_float();
    if (std::all_of(data->begin(), data->end(),
                    [](auto &x) { return x == floor(x); })) {
      std::vector<int64_t> data_v(data->begin(), data->end());
      input_shapes_v.push_back(data_v);
    }
  }
  if (input_shapes_v.size() == 3) {
    llvm_unreachable("not support for input_shapes_v ==3");
  }
  llvm_unreachable("not finished");
}
void ops::WhereOp::type_inference(){
  llvm_unreachable("not finished");
}
