#include "support/module.h"

void ops::ReduceOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto num_dims = in_shape.size();
  auto axes = module::getI64Array(getAxes());
  std::vector<int64_t> out_shape;
  bool fixed = false;
  for (auto &idx : *axes) {
    if (idx < 0) {
      idx += num_dims;
      fixed = true;
    }
  }
  if (axes->size() == 0) {
    // for onnx whithout axes attr
    axes->resize(num_dims);
    std::iota(axes->begin(), axes->end(), 0);
    fixed = true;
  }
  if (fixed) {
    Builder builder(getContext());
    setAxesAttr(builder.getI64ArrayAttr(*axes));
  }
  for (int i = 0; i < num_dims; i++) {
    if (std::find(axes->begin(), axes->end(), i) != axes->end()) {
      if (getKeepdims()) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(in_shape[i]);
    }
  }
  /* keepdims = false, reduce at all axis,
    it need to set the shape to [1] */
  if (!out_shape.size())
    out_shape.push_back(1);
  module::setShapeOrVerify(getOutput(), out_shape);
}
