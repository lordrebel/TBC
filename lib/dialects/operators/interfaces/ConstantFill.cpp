//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"



void ops::ConstantFillOp::shape_inference() {
  if (module::isWeight(getInput())) {
    auto weight = cast<ops::WeightOp>(getInput().getDefiningOp());
    const auto shape = weight.read<float>();
    std::vector<int64_t> shape_(shape->begin(), shape->end());
    int idx = 0;
    for(auto a : shape_) {
      if(a == -1)
        shape_[idx] = (int64_t)1;
      idx += 1;
    }
    module::setShapeOrVerify(getOutput(), shape_);
  } else if (module::isShape(getInput())) {
    auto out_shape = module::getShapeTensorValue(getInput());
    module::setShapeOrVerify(getOutput(), out_shape);
  }
}
