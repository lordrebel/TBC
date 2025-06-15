
#include "support/module.h"
#include "support/mathutil.h"

pool_attr_t ops::AvgPoolOp::parseParam() {
  pool_attr_t p = {0};
  auto ishape = dyn_cast<RankedTensorType>(getInput().getType()).getShape();
  auto oshape = dyn_cast<RankedTensorType>(getOutput().getType()).getShape();
  auto kernel = module::getI64Array(getKernelShape());
  auto stride = module::getI64Array(getStrides());
  auto pad = module::getI64Array(getPads());
  if (getKernelShape().size() == 3) {
    p.n = ishape[0];
    p.c = ishape[1];
    p.id = ishape[2];
    p.ih = ishape[3];
    p.iw = ishape[4];
    p.od = oshape[2];
    p.oh = oshape[3];
    p.ow = oshape[4];
    p.kd = kernel->at(0);
    p.kh = kernel->at(1);
    p.kw = kernel->at(2);
    p.sd = stride->at(0);
    p.sh = stride->at(1);
    p.sw = stride->at(2);
    p.pad_d = pad->at(0);
    p.pad_h = pad->at(1);
    p.pad_w = pad->at(2);
    p.pad_d_after = pad->at(3);
    p.pad_h_after = pad->at(4);
    p.pad_w_after = pad->at(5);
  } else if (getKernelShape().size() == 2) {
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
  } else if (getKernelShape().size() == 1) {
    p.id = 1;
    p.od = 1;
    p.kd = 1;
    p.kw = 1;
    p.sd = 1;
    p.sw = 1;
    module::getNCHW(ishape, p.n, p.c, p.ih, p.iw);
    module::getNCHW(oshape, p.n, p.c, p.oh, p.ow);
    p.kh = kernel->at(0);
    p.sh = stride->at(0);
    p.pad_h = pad->at(0);
    p.pad_h_after = pad->at(1);
  }
  p.pad_value = getPadValue();
  p.is_global = p.id == p.kd && p.ih == p.kh && p.iw == p.kw && p.od == 1 &&
                p.oh == 1 && p.ow == 1;
  p.count_include_pad = getCountIncludePad();
  return p;
}


void ops::AvgPoolOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  auto kernel_shape = module::getI64Array(getKernelShape());
  if (kernel_shape->size() == 0) {
    // for onnx GlobalAvgPool
    auto num_dim = input_shape.size() - 2;
    assert(num_dim > 0);
    std::vector<int64_t> vkernel_shape;
    std::vector<int64_t> vstrides(num_dim, 1);
    std::vector<int64_t> vpads(2 * num_dim, 0);
    for(uint32_t i = 2; i < input_shape.size(); i++) {
      vkernel_shape.push_back(input_shape[i]);
    }
    auto builder = OpBuilder(getContext());
    setKernelShapeAttr(builder.getI64ArrayAttr(vkernel_shape));
    setStridesAttr(builder.getI64ArrayAttr(vstrides));
    setPadsAttr(builder.getI64ArrayAttr(vpads));
    kernel_shape = module::getI64Array(getKernelShape());
  }
  assert(input_shape.size() > 2);
  int spacial_rank = input_shape.size() - 2;
  assert(spacial_rank == getKernelShape().size());
  assert(getPads().size() == spacial_rank * 2);
  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(input_shape[0]);
  out_shape.push_back(input_shape[1]);
  auto input_spacial_shape = llvm::ArrayRef(&input_shape[2], spacial_rank);
  auto pads = module::getI64Array(getPads());
  auto strides = module::getI64Array(getStrides());
  // for AutoPad
  std::vector<int64_t> new_pads(pads->begin(), pads->end());
  if (getAutoPad().has_value()) {
    set_auto_pad(getAutoPad().value(), input_shape, *kernel_shape, *strides,
                 new_pads);
    removeAutoPadAttr();
  }
  // for CeilMode
  if (getCeilMode().has_value() && getCeilMode().value()) {
    for(uint32_t i = 0; i <= 1; i++) {
      auto remain_pixel = (input_shape[i + 2] + 2 * new_pads[i] - kernel_shape->at(i)) % strides->at(i);
      if (remain_pixel > 0) {
        new_pads[i + 2] += (strides->at(i) - remain_pixel);
      }
    }
    removeCeilModeAttr();
  }
  auto builder = OpBuilder(getContext());
  setPadsAttr(builder.getI64ArrayAttr(new_pads));
  pads = module::getI64Array(getPads());

  for (int i = 0; i < spacial_rank; i++) {
    auto out_dim = (input_spacial_shape[i] + pads->at(i) +
                    pads->at(i + spacial_rank) - kernel_shape->at(i)) /
                       strides->at(i) +
                   1;
    out_shape.push_back(out_dim);
  }
  if (getKeepdims() == false) {
    while (out_shape.size() > 2) {
      if (out_shape.back() == 1) {
        out_shape.pop_back();
      } else {
        break;
      }
    }
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}
