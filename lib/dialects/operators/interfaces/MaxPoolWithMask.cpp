//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "float.h"
#include "support/module.h"

pool_attr_t ops::MaxPoolWithMaskOp::parseParam() {
  pool_attr_t p = {0};
  assert(getKernelShape().size() == 2); // only support 2d now
  auto ishape = dyn_cast<RankedTensorType>(getInput().getType()).getShape();
  auto oshape =dyn_cast<RankedTensorType>( getOutput().getType()).getShape();
  auto kernel = module::getI64Array(getKernelShape());
  auto stride = module::getI64Array(getStrides());
  auto pad = module::getI64Array(getPads());
  p.id = 1;
  p.od = 1;
  p.kd = 1;
  p.sd = 1;
  module::getNCHW(ishape, p.n, p.c, p.ih, p.iw);
  module::getNCHW(oshape, p.n, p.c, p.oh, p.ow);
  p.kh = kernel->at(0);
  p.kw = kernel->at(1);
  p.sh = stride->at(0);
  p.sw = stride->at(1);
  p.pad_h = pad->at(0);
  p.pad_w = pad->at(1);
  p.pad_h_after = pad->at(2);
  p.pad_w_after = pad->at(3);
  p.pad_value = 0;
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.is_global = p.id == p.kd && p.ih == p.kh && p.iw == p.kw && p.od == 1 &&
                p.oh == 1 && p.ow == 1;
  p.count_include_pad = getCountIncludePad();
  return p;
}

void ops::MaxPoolWithMaskOp::shape_inference() {
  int64_t out_h, out_w;
  auto input_shape = module::getShape(getInput());
  auto kernel = module::getI64Array(getKernelShape());
  auto stride = module::getI64Array(getStrides());
  auto pad = module::getI64Array(getPads());

  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(input_shape[0]);
  out_shape.push_back(input_shape[1]);
  out_h =
      (ceil(input_shape[2] + 2 * pad->at(0) - kernel->at(0)) / stride->at(0)) +
      1;
  out_w =
      (ceil(input_shape[3] + 2 * pad->at(1) - kernel->at(1)) / stride->at(1)) +
      1;
  out_shape.push_back(out_h);
  out_shape.push_back(out_w);

  module::setShapeOrVerify(getOutput(), out_shape);
  module::setShapeOrVerify(getMask(), out_shape);
}
