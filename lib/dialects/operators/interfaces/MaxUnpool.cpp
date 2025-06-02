//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

void ops::MaxUnpoolOp::shape_inference() {
  int64_t N, C, H, W;
  module::getNCHW(getInput(), N, C, H, W);
  auto scale_h_ = getScaleH();
  auto scale_w_ = getScaleW();
  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(N);
  out_shape.push_back(C);
  out_shape.push_back(scale_h_ * H);
  out_shape.push_back(scale_w_ * W);
  module::setShapeOrVerify(getOutput(), out_shape);
}
