//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

void ops::LayerNormOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto dims = in_shape.size();
  auto axis = getAxis();
  if (axis < 0) {
    axis += dims;
    setAxis(axis);
  }
  auto normalized_shape = module::getI64Array(getNormalizedShape());
  if (normalized_shape->size() == 0) {
    for (uint32_t i = axis; i < dims; i++) {
      normalized_shape->push_back(in_shape[i]);
    }
    auto builder = OpBuilder(getContext());
    setNormalizedShapeAttr(builder.getI64ArrayAttr(*normalized_shape));
  }
  if (!std::equal(normalized_shape->begin(), normalized_shape->end(),
                  in_shape.begin() + axis)) {
    dump();
    llvm_unreachable("normalized_shape is illegal");
  }
  module::setShapeOrVerify(getOutput(), in_shape);

  if (module::isWeight(getWeight())) {
    broadcast_tensor_reshape(getOutput(), getWeight());
  }
  if (module::isWeight(getBias())) {
    broadcast_tensor_reshape(getOutput(), getBias());
  }

}
