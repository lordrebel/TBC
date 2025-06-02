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


void ops::CompareConstOp::shape_inference() {
  common_shape_inference(getOperation());
  if (module::isShape(getInput())) {
    llvm_unreachable("not support shape inference for compare const op yet. ");
  }
}
