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


void ops::TileOp::shape_inference() {
  auto in0_shape = module::getShape(getInput());
  std::vector<int64_t> tile_vec;
  if (getTile().has_value()){
      auto tile_v = module::getI64Array(getTile().value());
      tile_vec = *tile_v;
  } else if (auto tile_w = dyn_cast<ops::WeightOp>(getTileT().getDefiningOp())){
    auto tile_v = tile_w.read_as_float();
    std::transform(tile_v->begin(), tile_v->end(),
        std::back_inserter(tile_vec),
        [](auto &v) { return static_cast<int64_t>(v); });
  } else if (module::isShape(getTileT())) {
      tile_vec = module::getShapeTensorValue(getTileT());
  } else{
    llvm_unreachable("tile_vec is illegal");
  }
  assert(in0_shape.size() == tile_vec.size());
  std::vector<int64_t> out_shape(in0_shape.size());
  std::transform(tile_vec.begin(), tile_vec.end(), in0_shape.begin(), out_shape.begin(),
        [](int a, int b){return a * b;});
  module::setShapeOrVerify(getOutput(), out_shape);
}
void ops::TileOp::type_inference() {
  auto input=getInput();
  auto output=getOutput();
  module::setElementType(output, module::getElementType(input));
}
