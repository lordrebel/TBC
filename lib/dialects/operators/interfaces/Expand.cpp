//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"


void ops::ExpandOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  std::vector<int64_t> out_shape;
  if (!getShapeT()){
      auto shape_v = module::getI64Array(getShape());
      out_shape = *shape_v;
  } else if (auto shape_w = dyn_cast<ops::WeightOp>(getShapeT().getDefiningOp())){
    auto shape_v = shape_w.read_as_float();
    std::transform(shape_v->begin(), shape_v->end(),
        std::back_inserter(out_shape),
        [](auto &v) { return static_cast<int64_t>(v); });
  } else if (module::isShape(getShapeT())) {
      out_shape = module::getShapeTensorValue(getShapeT());
  } else{
    llvm_unreachable("out_shape is illegal");
  }

  int dim_in = in_shape.size();
  int dim_out = out_shape.size();
  int dim_pad = dim_out - dim_in;
  assert(dim_pad >= 0);

  for(int i = dim_pad; i < dim_out; i++){
    out_shape[i] = in_shape[i - dim_pad] == 1 ? out_shape[i] : in_shape[i - dim_pad];
  }
  module::setShapeOrVerify(getOutput(), out_shape);

  if (module::isShape(getInput())) {
    module::bindShapeTensorValue(getOutput(), out_shape);
  }

}
void ops::ExpandOp::type_inference() {
  common_type_inference(getOperation());
}
