//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

void ops::GridSamplerOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  auto grid_shape = module::getShape(getGrid());
  auto dims = grid_shape.size();
  std::vector<int64_t> out_shape;
  out_shape.push_back(input_shape[0]);
  out_shape.push_back(input_shape[1]);
  out_shape.push_back(grid_shape[1]);
  out_shape.push_back(grid_shape[2]);
  if (dims > 4)
    out_shape.push_back(grid_shape[3]);
  auto out = getOutput();
  module::setShapeOrVerify(out, out_shape);

  // unsqueeze grid shape
  if (grid_shape.size() != input_shape.size() &&
      grid_shape[grid_shape.size() - 1] == 1) {
    std::vector<int64_t> new_shape(grid_shape.begin(),
                                   grid_shape.begin() + input_shape.size());
    module::setShape(getGrid(), new_shape);
  }
}
