//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"

// clang-format off
// case 1: [5, 6] * [6, 7] = [5, 7] => batch = 1, M = 5, K = 6, N = 7
// case 2: [1, 512, 7, 7] * [25088, 4096] = [1, 4096] => batch = 1, M = 1, K = 25088, N = 4096
// case 3: [3, 4, 5, 6] * [3, 4, 6, 7] = [3, 4, 5, 7] => batch = 12, M = 5, K = 6, N = 7
// case 4: [4, 5, 6] * [6,7] = [4, 5, 7] => batch =1, M = 20, K = 6, N = 7
// case 5: [4, 5, 6] * [6] = [4, 5] => batch =1, M = 20, K = 6, N = 1
// case 6: [4096] * [4096, 12884] = [1,12884] => batch =1, M = 1, K = 4096, N = 12884
// clang-format on
matmul_attr_t ops::MatMulOp::parseParam() {
  matmul_attr_t p = {0};
  auto a_s = SmallVector<int64_t>(module::getShape(getLeft()));
  auto b_s = SmallVector<int64_t>(module::getShape(getRight()));
  auto o_s = SmallVector<int64_t>(module::getShape(getOutput()));
  p.with_bias = !module::isNone(getBias());

  auto a_dims = a_s.size();
  auto b_dims = b_s.size();
  auto o_dims = o_s.size();
  p.right_transpose =false;
  if (b_dims == 1) {
    assert(p.right_transpose == false);
    b_s.push_back(1);
    o_s.push_back(1);
    b_dims += 1;
    o_dims += 1;
  }
  if (a_dims == 1) {
    assert(p.left_transpose == false);
    a_s.insert(a_s.begin(), 1);
    o_s.insert(o_s.begin(), 1);
    a_dims += 1;
    o_dims += 1;
  }
  p.N = p.right_transpose ? b_s[b_dims - 2] : b_s[b_dims - 1];
  assert(p.N == o_s[o_dims - 1]);
  p.K = p.right_transpose ? b_s[b_dims - 1] : b_s[b_dims - 2];
  p.batch = 1;
  int b_temp = 1;
  for (int i = 0; i < b_dims - 2; i++) {
    p.batch *= b_s[i];
  }
  for (int i = 1; i < b_dims - 2; i++) {
    b_temp *= b_s[i];
  }
  if (a_s[0] == b_s[0] && b_temp == 1 && b_dims > 2){
    p.batch = b_s[0];
    int a_temp = 1;
    for(int i = 1; i < a_dims - 2; i++){
      a_temp *= a_s[i];
    }
    p.M = a_s[o_dims - 2] * a_temp;
  } else {
    if (p.batch > 1 || o_dims <= 2) {
      p.M = o_s[o_dims - 2];
    } else {
      p.M = std::accumulate(o_s.begin(), o_s.begin() + o_dims - 1, 1,
                            std::multiplies<int64_t>());
    }
  }
  return p;
}


// shape case:
// case 1: [5, 6] * [6, 7] = [5, 7]
// case 2: [1, 512, 7, 7] * [25088, 4096] = [1, 4096]
// case 3: [3, 4, 5, 6] * [3, 4, 6, 7] = [3, 4, 5, 7]
// case 4: [4, 5, 6] * [6, 7] = [4, 5, 7]
// case 5: [4, 5, 6] * [6] = [4, 5]
// case 6: keep_dims == false, [4, 5, 6] * [6, 7] = [20, 7]
void ops::MatMulOp::shape_inference() {
  std::vector<int64_t> in0_shape = module::getShape(getLeft());
  int in0_dims = in0_shape.size();
  auto k = in0_shape[in0_dims - 1];
  std::vector<int64_t> in1_shape = module::getShape(getRight());
  int in1_dims = in1_shape.size();
  bool r_transpose = false;
  bool keep_dims_ = getKeepDims();
  int k_idx = in1_dims - (r_transpose ? 1 : 2);
  int n_idx = in1_dims - (r_transpose ? 2 : 1);
  auto n = in1_shape[n_idx];
  std::vector<int64_t> out_shape = in0_shape;
  if (in1_dims == 1) {
    assert(in1_shape[0] == k);
    out_shape.pop_back();
  } else if (in1_shape[k_idx] == k) {

      out_shape[in0_dims - 1] = n;

  } else if (in1_dims == 2) {
    auto sum = in1_shape[k_idx];
    while (out_shape.size() > 0 && sum % out_shape.back() == 0 && sum != 1) {
      sum = sum / out_shape.back();
      out_shape.pop_back();
    }
    if (sum != 1) {
      dump();
      llvm_unreachable("shape is illegal");
    }
    out_shape.push_back(n);
  } else {
    out_shape[in0_dims - 1] = n;
  }
  if (!keep_dims_) {
    int64_t batch_size = std::accumulate(out_shape.begin(), out_shape.end() - 1,
                                         1, std::multiplies<int64_t>());
    out_shape.resize(2);
    out_shape[0] = batch_size;
    out_shape[1] = n;
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}
