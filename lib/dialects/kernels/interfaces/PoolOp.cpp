#include"dialects/kernels/IR/kernels.h"
#include "support/module.h"
pool_attr_t tbc::kls::Pool1DOp::parseParam() {
  pool_attr_t p = {0};
  p.id = 1;
  p.od = 1;
  p.kd = 1;
  p.kw = 1;
  p.sd = 1;
  p.sw = 1;
  auto ishape = dyn_cast<RankedTensorType>(getInput().getType()).getShape();
  auto oshape = dyn_cast<RankedTensorType>(getOutput().getType()).getShape();
  module::getNCHW(ishape, p.n, p.c, p.ih, p.iw);
  module::getNCHW(oshape, p.n, p.c, p.oh, p.ow);
  assert(p.iw == 1 && p.ow == 1);

  auto kernel = module::getI64Array(getKernelShape());
  p.kh = kernel->at(0);
  auto stride = module::getI64Array(getStrides());
  p.sh = stride->at(0);
  auto pad = module::getI64Array(getPads());
  p.pad_h = pad->at(0);
  p.pad_h_after = pad->at(1);
  p.pad_value = getPadValue();
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.is_global = p.ih == p.kh && p.iw == p.kw && p.oh == 1 && p.ow == 1;
  return p;
}


pool_attr_t tbc::kls::Pool2DOp::parseParam() {
  pool_attr_t p = {0};
  p.id = 1;
  p.od = 1;
  p.kd = 1;
  p.sd = 1;
  auto ishape = dyn_cast<RankedTensorType>(getInput().getType()).getShape();
  auto oshape = dyn_cast<RankedTensorType>(getOutput().getType()).getShape();
  module::getNCHW(ishape, p.n, p.c, p.ih, p.iw);
  module::getNCHW(oshape, p.n, p.c, p.oh, p.ow);

  auto kernel = module::getI64Array(getKernelShape());
  p.kh = kernel->at(0);
  p.kw = kernel->at(1);
  auto stride = module::getI64Array(getStrides());
  p.sh = stride->at(0);
  p.sw = stride->at(1);
  auto pad = module::getI64Array(getPads());
  p.pad_h = pad->at(0);
  p.pad_w = pad->at(1);
  p.pad_h_after = pad->at(2);
  p.pad_w_after = pad->at(3);
  p.pad_value = getPadValue();
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.is_global = p.ih == p.kh && p.iw == p.kw && p.oh == 1 && p.ow == 1;
  return p;
}
