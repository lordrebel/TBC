//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

void ops::Depth2SpaceOp::shape_inference() {
  int64_t in, ic, ih, iw, oc, oh, ow;
  if (getInIs_NCHW()) {
    module::getNCHW(getInput(), in, ic, ih, iw, false);
  } else {
    module::getNCHW(getInput(), in, ih, iw, ic, false);
  }
  auto in_shape = module::getShape(getInput());
  auto num_dims = in_shape.size();
  std::vector<int64_t> out_shape = in_shape;
  auto block_h = getBlockH();
  auto block_w = getBlockW();
  if (getIsInversed()) {
    oc = ic * block_h * block_w;
    oh = ih / block_h;
    ow = iw / block_w;
  } else {
    oc = ic / (block_h * block_w);
    oh = ih * block_h;
    ow = iw * block_w;
  }
  if (getOutIs_NCHW()) {
    out_shape[num_dims - 3] = oc;
    out_shape[num_dims - 2] = oh;
    out_shape[num_dims - 1] = ow;
  } else {
    out_shape[num_dims - 3] = oh;
    out_shape[num_dims - 2] = ow;
    out_shape[num_dims - 1] = oc;
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}
