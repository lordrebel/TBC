//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "interfaces/typeInfer_interface.h"
#include "support/module.h"

void ops::PadOp::shape_inference() {
  auto pads_origin = module::getI64Array(getPaddings());
  auto input_shape = module::getShape(getInput());
  auto dim = input_shape.size();
  std::vector<int64_t> pads(dim * 2, 0);
  if (module::isPlatform(tbc::utils::Platform::TORCH)) {
    if (pads_origin->size() >= 2) {
      // w pad
      pads[dim - 1] = pads_origin->at(0);
      pads[2 * dim - 1] = pads_origin->at(1);
    }
    if (pads_origin->size() >= 4) {
      pads[dim - 2] = pads_origin->at(2);
      pads[dim * 2 - 2] = pads_origin->at(3);
    }
    Builder builder(getContext());
    setPaddingsAttr(builder.getI64ArrayAttr(pads));
  } else {
    assert(pads_origin->size() == dim * 2);
    pads = *pads_origin;
  }
  std::vector<int64_t> out_shape(input_shape);
  for (int i = 0; i < dim; i++)
    out_shape[i] += pads[i] + pads[i + dim];
  module::setShapeOrVerify(getOutput(), out_shape);
}
void ops::PadOp::type_inference() {
  common_type_inference(getOperation());
}
