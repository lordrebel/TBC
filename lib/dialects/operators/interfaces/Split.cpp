//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

void ops::SplitOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto num = getNum();
  auto axis = getAxis();
  if (axis < 0) {
    axis += in_shape.size();
    setAxis(axis);
  }
  int64_t out_max_size = (in_shape[axis] + num - 1) / num;
  auto split_size = module::getI64Array(getSplitSize(), num, out_max_size);
  std::vector<int64_t> out_size(*split_size);
  auto length = std::accumulate(out_size.begin(), out_size.end(), 0);
  out_size[num - 1] = out_size[num - 1] + in_shape[axis] - length;
  OpBuilder builder(module::getCtx());
  setSplitSizeAttr(builder.getI64ArrayAttr(out_size));
  assert(num == getOutputs().size());
  std::vector<int64_t> out_shape = in_shape;

  for (int i = 0; i < num; ++i) {
    auto out = getResult(i);
    out_shape[axis] = out_size[i];
    module::setShapeOrVerify(out, out_shape);
  }
}
