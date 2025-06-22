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


void ops::CastOp::shape_inference() { common_shape_inference(getOperation()); }
void ops::CastOp::type_inference() {
  //TODO: finish this with converter
  llvm_unreachable("not implemented yet");
}
