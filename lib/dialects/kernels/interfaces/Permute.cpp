#include"dialects/kernels/IR/Kernels.h"
#include "support/mathutil.h"
#include "support/module.h"
permute_attr_t tbc::kls::PermuteOp::parseParam() {
  permute_attr_t attr;
  std::vector<int64_t> in_shape = module::getShape(getInput());
  i64_array_t in_order = module::getI64Array(getOrder());
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
