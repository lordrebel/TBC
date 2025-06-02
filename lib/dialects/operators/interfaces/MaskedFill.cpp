//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

void ops::MaskedFillOp::shape_inference() {
  broadcast_shape_inference(getOperation());
  for(int i = 0; i< getNumOperands(); ++i) {
    auto value = getOperation()->getOperand(i);
    broadcast_tensor_reshape(getOutput(), value);
  }
}
