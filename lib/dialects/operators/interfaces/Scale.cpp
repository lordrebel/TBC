//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "interfaces/typeInfer_interface.h"
#include "support/module.h"


void ops::ScaleOp::shape_inference() { common_shape_inference(getOperation()); }

void ops::ScaleOp::type_inference() { broadcast_type_inference(getOperation()); }
