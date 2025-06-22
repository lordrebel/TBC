//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"
#include"support/mathutil.h"
#include "llvm/Support/ErrorHandling.h"
#include <valarray>

void ops::SliceOp::paramConvert() {
  auto context = getContext();
  mlir::Builder builder(context);
  auto offset_ori = module::getI64Array(getOffset());
  auto steps_ori = module::getI64Array(getSteps());
  auto ends_ori = module::getI64Array(getEnds());
  auto axes_ori = module::getI64Array(getAxes());
  auto input_shapes = module::getShape(getInput());

  auto input_dims = input_shapes.size();
  auto slice_n = axes_ori->size();
  assert(offset_ori->size() == slice_n && steps_ori->size() == slice_n &&
         ends_ori->size() == slice_n);
  auto offset_v = std::make_shared<std::vector<int64_t>>(input_dims, 0);
  auto steps_v = std::make_shared<std::vector<int64_t>>(input_dims, 1);
  auto ends_v = std::make_shared<std::vector<int64_t>>(input_shapes);
  for (int i = 0; i < slice_n; ++i) {
    int axis =
        axes_ori->at(i) >= 0 ? axes_ori->at(i) : axes_ori->at(i) + input_dims;
    int step = steps_ori->at(i);
    int64_t end = ends_ori->at(i) >= 0 ? ends_ori->at(i)
                                       : ends_ori->at(i) + input_shapes[axis];
    end = step > 0 ? std::clamp(end, 0L, input_shapes[axis])
                   : std::clamp(end, -1L, input_shapes[axis] - 1);
    int64_t offset = offset_ori->at(i) >= 0
                         ? offset_ori->at(i)
                         : offset_ori->at(i) + input_shapes[axis];
    offset = step > 0 ? std::clamp(offset, 0L, input_shapes[axis])
                      : std::clamp(offset, 0L, input_shapes[axis] - 1);
    offset_v->at(axis) = offset;
    ends_v->at(axis) = end;
    steps_v->at(axis) = step;
  }
  setOffsetAttr(builder.getI64ArrayAttr(*offset_v));
  setStepsAttr(builder.getI64ArrayAttr(*steps_v));
  setEndsAttr(builder.getI64ArrayAttr(*ends_v));
  setAxesAttr(builder.getI64ArrayAttr(std::nullopt));
}
void ops::SliceOp::shape_inference() {
  if (!getAxes().empty())
    paramConvert();
  const auto input_shape = module::getShape(getInput());
  const size_t dims = input_shape.size();
  const auto offset_v = module::getI64Array(getOffset());
  const auto steps_v = module::getI64Array(getSteps());
  const auto ends_v = module::getI64Array(getEnds());
  const size_t slice_dims = offset_v->size();
  std::vector<int64_t> output_shape(input_shape.size());
  for (size_t i = 0; i < dims; ++i) {
    if (i < slice_dims) {
      if (ends_v->at(i) == -1) {
        output_shape[i] = input_shape[i];
        ends_v->at(i) = output_shape[i];
      } else
        output_shape[i] =
            abs_ceiling_func(ends_v->at(i) - offset_v->at(i), steps_v->at(i));
    } else {
      output_shape[i] = input_shape[i];
    }
  }
  module::setShapeOrVerify(getOutput(), output_shape);

  if (module::isShape(getInput())) {
   llvm_unreachable("not support shape op in shape inference yet, ");
  }
}
void ops::SliceOp::type_inference() {
  auto input=getInput();
  auto output=getOutput();
  module::setElementType(output, module::getElementType(input));
}
