//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"
#include "support/mathutil.h"


void ops::RepeatOp::shape_inference() {
  assert(module::isWeight(getRepeats()));
  auto repeat_op = getRepeats().getDefiningOp<ops::WeightOp>();
  auto repeats = repeat_op.read<float>();
  auto in_shape = module::getShape(getInput());
  int64_t dim = std::max(in_shape.size(), (*repeats).size());
  auto in_shape_ = shape_expand_dim(in_shape, dim);
  auto repeats_ = shape_expand_dim(*repeats, dim);
  auto out_shape = llvm::SmallVector<int64_t>();
  for (int i = 0; i < dim; ++i) {
    out_shape.push_back(in_shape_[i] * repeats_[i]);
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}

void ops::RepeatOp::type_inference() {
  auto input=getInput();
  auto output=getOutput();
  module::setElementType(output, module::getElementType(input));

}
