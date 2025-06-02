//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

using namespace std;


void ops::GatherNDOp::shape_inference() {
  auto batch_dims = getBatchDims();
  auto data_rank = module::getShape(getInput()).size();
  auto indices_shape = module::getShape(getIndices());
  auto input_shape = module::getShape(getInput());
  std::vector<int64_t> output_shape;
  for (int i = 0; i < batch_dims; ++i) {
    output_shape.push_back(indices_shape[i]);
  }
  for (int i = batch_dims; i < indices_shape.size() - 1; ++i) {
    output_shape.push_back(indices_shape[i]);
  }
  if (indices_shape[indices_shape.size() - 1] != data_rank - batch_dims) {
    for (int i = batch_dims + indices_shape[indices_shape.size() - 1];
         i < input_shape.size(); ++i) {
      output_shape.push_back(input_shape[i]);
    }
  }
  module::setShapeOrVerify(getOutput(), output_shape);
}
