#include "dialects/kernels/IR/kernels.h"
#include "support/module.h"

using namespace mlir;
namespace tbc::kls {
conv_attr_t Conv2dOp::parseParam() {
  conv_attr_t p = {0};
  auto i_s = cast<RankedTensorType>(getInput().getType()).getShape();
  auto o_s = cast<RankedTensorType>(getOutput().getType()).getShape();
  p.has_bias = !isa<NoneType>(getBias().getType());

  auto kernel = module::getI64Array(getKernelShape());
  auto pads_v = module::getI64Array(getPads());
  auto strides_v = module::getI64Array(getStrides());
  auto dilation = module::getI64Array(getDilations(), kernel->size(), 1);
  p.n = i_s[0];
  p.ic = i_s[1];
  p.oc = o_s[1];
  p.dims = i_s.size() - 2;
  // 2d conv
  p.id = p.od = p.kd = p.dd = p.sd = 1;
  p.ih = i_s[2];
  p.iw = i_s[3];
  p.oh = o_s[2];
  p.ow = o_s[3];
  p.kh = kernel->at(0);
  p.kw = kernel->at(1);
  p.pht = pads_v->at(0);
  p.pwl = pads_v->at(1);
  p.phb = pads_v->at(2);
  p.pwr = pads_v->at(3);
  p.sh = strides_v->at(0);
  p.sw = strides_v->at(1);
  p.dh = dilation->at(0);
  p.dw = dilation->at(1);
  p.groups = getGroup();
  p.is_dw = (p.oc == p.ic && p.oc == p.groups && p.groups > 1);
  return p;
}
} // namespace tbc::kls
