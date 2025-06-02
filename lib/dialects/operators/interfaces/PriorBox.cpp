//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"


void ops::PriorBoxOp::shape_inference() {
  int64_t num_priors = getNumPriors();
  auto input_shape = module::getShape(getInputs()[0]);
  int layer_height = input_shape[2];
  int layer_width = input_shape[3];
  llvm::SmallVector<int64_t> out_shape;
  int dim = layer_height * layer_width * num_priors * 4;
  out_shape.push_back(1);
  out_shape.push_back(2);
  out_shape.push_back(dim);
  module::setShapeOrVerify(getOutput(), out_shape);
}
