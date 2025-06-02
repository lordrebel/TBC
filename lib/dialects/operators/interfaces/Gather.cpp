//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"
#include "llvm/Support/ErrorHandling.h"

void ops::GatherOp::shape_inference() {
  auto indices_shape = module::getShape(getIndices());
  auto ax = getAxis();
  auto input_shape = module::getShape(getInput());
  if (ax < 0) {
    ax += input_shape.size();
    setAxis(ax);
  }
  std::vector<int64_t> out_shape;
  for (int i = 0; i < ax; ++i) {
    out_shape.push_back(input_shape[i]);
  }

  if (indices_shape.size() == 1 && indices_shape[0] == 1 && !getKeepdims()) {
    // if indices_shape.size() == 1 and indices is scalar(not a array) do
    // squeeze manner do nothing
  } else {
    for (int s : indices_shape) {
      out_shape.push_back(s);
    }
  }
  for (int i = ax + 1; i < input_shape.size(); ++i) {
    out_shape.push_back(input_shape[i]);
  }
  if (out_shape.size() == input_shape.size()) {
    auto builder = OpBuilder(getContext());
    setKeepdimsAttr(builder.getBoolAttr(true));
  }
  module::setShapeOrVerify(getOutput(), out_shape);
  if (module::isShape(getInput())) {
    llvm_unreachable("not implemented for shape inference of gather op with input shape");
  }
}
