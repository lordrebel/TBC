//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

// RangeOp is special, will convert to WeightOp
void ops::RangeOp::shape_inference() {
  int64_t start = 0, delta = 1, limit = 0;
  if (!module::isNone(getStart())) {
    if (auto start_w = dyn_cast<ops::WeightOp>(getStart().getDefiningOp())) {
      auto start_v = start_w.read_as_float();
      assert(start_v->size() == 1);
      start = static_cast<float>(start_v->at(0));
    } else if (module::isShape(getStart())) {
      auto start_v = module::getShapeTensorValue(getStart());
      assert(start_v.size() == 1);
      start = start_v[0];
    } else {
      llvm_unreachable("start must be a weight or a shape");
    }
  }
  if (!module::isNone(getDelta())) {
    if (auto delta_w = dyn_cast<ops::WeightOp>(getDelta().getDefiningOp())) {
      auto delta_v = delta_w.read_as_float();
      assert(delta_v->size() == 1);
      delta = static_cast<float>(delta_v->at(0));
    } else if (module::isShape(getDelta())) {
      auto delta_v = module::getShapeTensorValue(getDelta());
      assert(delta_v.size() == 1);
      delta = delta_v[0];
    } else {
      llvm_unreachable("delta must be a weight or a shape");
    }
  }
  if (auto limit_w = dyn_cast<ops::WeightOp>(getLimit().getDefiningOp())) {
    auto limit_v = limit_w.read_as_float();
    assert(limit_v->size() == 1);
    limit = static_cast<float>(limit_v->at(0));
  } else if (module::isShape(getLimit())) {
    auto limit_v = module::getShapeTensorValue(getLimit());
    assert(limit_v.size() == 1);
    limit = limit_v[0];
  } else {
    llvm_unreachable("delta must be a weight or a shape");
  }
  auto out_size = (limit - start) / delta;
  module::setShapeOrVerify(getOutput(), {out_size});
}
void ops::RangeOp::type_inference() {
  broadcast_type_inference(getOperation());
}
