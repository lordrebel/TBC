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

void ops::CompareOp::shape_inference() {
  broadcast_shape_inference(getOperation());
  for (int i = 0; i < getNumOperands(); i++) {
    auto value = getOperation()->getOperand(i);
    broadcast_tensor_reshape(getOutput(), value);
  }
  auto inputs = {getLhs(), getRhs()};
  // shape value inference can only support shape and weight
  bool need_shape_val_infer =
      std::all_of(inputs.begin(), inputs.end(), [](auto in_op) {
        return module::isShape(in_op) || module::isWeight(in_op);
      });
  if (need_shape_val_infer) {
    llvm_unreachable("not support shape value inference for compare op yet. ");
  }
}

void ops::CompareOp::type_inference() {
  auto op=getOperation();
  auto boolType=IntegerType::get(op->getContext(), 1);
  module::setElementType(this->getOutput(), boolType);
}
