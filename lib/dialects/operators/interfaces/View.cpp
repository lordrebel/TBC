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
#include "llvm/Support/raw_ostream.h"


void ops::ViewOp::shape_inference() {
  auto weight = cast<ops::WeightOp>(getShape().getDefiningOp());
  auto shape = weight.read_as_float();

  std::vector<int64_t> shape_(shape->begin(), shape->end());
  auto op = getOperation();
  OpBuilder builder(module::getCtx());
  builder.setInsertionPointAfter(op);
  auto out = getOutput();
  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(
      builder.getNamedAttr("shape", builder.getI64ArrayAttr(shape_)));
  auto new_op = builder.create<ops::ReshapeOp>(
      getLoc(), out.getType(), ArrayRef<Value>{getInput()}, attrs);
  out.replaceAllUsesWith(new_op.getOutput());
  new_op.shape_inference();
}
void ops::ViewOp::type_inference() {
  auto input=getInput();
  auto output=getOutput();
  module::setElementType(output, module::getElementType(input));
}
