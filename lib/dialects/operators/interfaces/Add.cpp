//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

void ops::AddOp::shape_inference() {
  broadcast_shape_inference(getOperation());
  for (int i = 0; i < getNumOperands(); i++) {
    auto value = getInputs()[i];
    broadcast_tensor_reshape(getOutput(), value);
  }
  auto inputs = getInputs();
  // shape value inference can only support shape and weight
  bool need_shape_val_infer =
      std::all_of(inputs.begin(), inputs.end(), [](auto in_op) {
        return module::isWeight(in_op) || module::isShape(in_op);
      }) &&
      std::any_of(inputs.begin(), inputs.end(), [](auto in_op) {
        return module::isShape(in_op);
      });
  if (need_shape_val_infer) {
    llvm_unreachable("Not Implemented for shape op inerence");
  }
}
