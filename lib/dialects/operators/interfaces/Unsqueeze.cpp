//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "dialects/operators/IR/operator.h"
#include "support/module.h"
#include "llvm/Support/ErrorHandling.h"



void ops::UnsqueezeOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto axes = module::getI64Array(getAxesAttr());
  std::vector<int64_t> out_shape(in_shape);
  std::vector<int64_t> axes_(*axes);
  int64_t out_dims = in_shape.size() + axes_.size();
  for (int i = 0; i < axes_.size(); ++i) {
    if (axes_[i] < 0) {
      axes_[i] += out_dims;
    }
  }
  std::sort(axes_.begin(), axes_.end());
  for (auto axis : axes_) {
    out_shape.insert(out_shape.begin() + axis, 1);
  }
  module::setShapeOrVerify(getOutput(), out_shape);
  if (module::isShape(getInput())) {
    llvm_unreachable("not supported shape inference for unsqueeze op yet");
  }
}

void ops::UnsqueezeOp::type_inference() {
  auto input=getInput();
  auto output=getOutput();
  module::setElementType(output, module::getElementType(input));
}
