

#include "support/module.h"


pool_attr_t ops::AdaptiveAvgPoolOp::parseParam() {
  llvm_unreachable("Not Implemented");
}

void ops::AdaptiveAvgPoolOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  auto output_size = module::getI64Array(getOutputSize());
  int spatial_rank = input_shape.size() - 2;
  auto input_spatial_shape = llvm::ArrayRef(&input_shape[2], spatial_rank);

  std::vector<int64_t> strides(spatial_rank, 0);
  std::vector<int64_t> kernel_shape(spatial_rank, 0);
  std::vector<int64_t> pads(2 * spatial_rank, 0);

  for (int i = 0; i < spatial_rank; i++) {
    if (output_size->at(i) == -1) {
      output_size->at(i) = input_spatial_shape[i];
    }
    strides[i] = std::floor(input_spatial_shape[i] / output_size->at(i));
    kernel_shape[i] =
        input_spatial_shape[i] - (output_size->at(i) - 1) * strides[i];
    strides[i] = output_size->at(i) == 1 ? 1 : strides[i];
  }

  auto op = getOperation();
  OpBuilder builder(module::getCtx());
  builder.setInsertionPointAfter(op);
  auto out = getOutput();
  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(builder.getNamedAttr(
      "kernel_shape", builder.getI64ArrayAttr(kernel_shape)));

  attrs.emplace_back(
      builder.getNamedAttr("strides", builder.getI64ArrayAttr(strides)));
  attrs.emplace_back(
      builder.getNamedAttr("pads", builder.getI64ArrayAttr(pads)));
  attrs.emplace_back(
      builder.getNamedAttr("count_include_pad", builder.getBoolAttr(true))
  );

  auto new_op = builder.create<ops::AvgPoolOp>(
      getLoc(), out.getType(), ArrayRef<Value>{getInput()}, attrs);
  out.replaceAllUsesWith(new_op.getOutput());
  new_op.shape_inference();

}
void ops::AdaptiveAvgPoolOp::type_inference() { common_type_inference(getOperation()); }
