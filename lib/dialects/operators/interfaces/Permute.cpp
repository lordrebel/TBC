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
permute_attr_t ops::PermuteOp::parseParam() {
  permute_attr_t attr;
  std::vector<int64_t> in_shape = module::getShape(getInput());
  i64_array_t in_order = module::getI64Array(getOrder());
  if (in_order->size() == 0) {
    // default revert it, eg: shape (2, 3, 4)->(4, 3, 2), per=[2, 1, 0]
    std::vector<int64_t> order;
    for(uint32_t i = in_shape.size() - 1; i >= 0; i--) {
      order.push_back(i);
    }
    auto builder = OpBuilder(getContext());
    setOrderAttr(builder.getI64ArrayAttr(order));
    in_order = module::getI64Array(getOrder());
  }
  auto ret =
      permute_reset(in_shape, *in_order, attr.in_shape_fix, attr.order_fix, 4);
  if (ret == false) {
    ret = permute_reset(in_shape, *in_order, attr.in_shape_fix, attr.order_fix,
                        5);
  }
  if (ret == false) {
    ret = permute_reset(in_shape, *in_order, attr.in_shape_fix, attr.order_fix,
                        6);
  }
  if (ret == false) {
    dump();
    llvm_unreachable("Not Implemented");
  }
  for (auto o : attr.order_fix) {
    attr.out_shape_fix.push_back(attr.in_shape_fix[o]);
  }
  return attr;
}


void ops::PermuteOp::shape_inference() {
  i64_array_t in_order = module::getI64Array(getOrder());
  auto in_shape = module::getShape(getInput());
  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < in_shape.size(); ++i) {
    out_shape.push_back(in_shape[(*in_order)[i]]);
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}
