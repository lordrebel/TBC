//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "dialects/operators/IR/operator.h"
#include "interfaces/typeInfer_interface.h"
#include "support/module.h"

void ops::Pow2Op::shape_inference() { common_shape_inference(getOperation()); }
void ops::Pow2Op::type_inference(){
  common_type_inference(getOperation());
}
