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

lstm_attr_t ops::LSTMOp::parseParam() {
  lstm_attr_t attr = {0};
  auto in_shape = module::getShape(getInput());
  assert(in_shape.size() == 3);
  if (getBatchFirst()) {
    attr.batch_size = in_shape[0];
    attr.seq_len = in_shape[1];
    attr.batch_first = true;
  } else {
    attr.batch_size = in_shape[1];
    attr.seq_len = in_shape[0];
    attr.batch_first = false;
  }
  attr.input_size = in_shape[2];
  attr.num_direction = getBidirectional() ? 2 : 1;
  attr.hidden_size = getHiddenSize();
  attr.have_bias = !isa<NoneType>(getBias().getType());
  attr.have_h0 = !isa<NoneType>(getInitialH().getType());
  attr.have_c0 = !isa<NoneType>(getInitialC().getType());
  attr.have_cont = !isa<NoneType>(getCont().getType());
  attr.output_y = !isa<NoneType>(getY().getType());
  attr.output_yh = !isa<NoneType>(getYH().getType());
  attr.output_yc = !isa<NoneType>(getYC().getType());
  return attr;
}

void ops::LSTMOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  assert(in_shape.size() == 3);
  int64_t num_dir = 1;
  if (getBidirectional()) {
    num_dir = 2;
  }
  int64_t seq_len, batch_size;
  if (getBatchFirst()) {
    batch_size = in_shape[0];
    seq_len = in_shape[1];
  } else {
    seq_len = in_shape[0];
    batch_size = in_shape[1];
  }
  int64_t hidden_size = getHiddenSize();
  std::vector<int64_t> shape0;
  std::vector<int64_t> shape1;
  std::vector<int64_t> shape2;
  if (getBatchFirst()) {
    shape0 = {batch_size, seq_len, num_dir, hidden_size};
    shape1 = {batch_size, num_dir, hidden_size};
    shape2 = {batch_size, num_dir, hidden_size};
  } else {
    if (module::isPlatform(tbc::utils::Platform::TORCH)) {
      shape0 = {seq_len, batch_size, num_dir, hidden_size};
    } else {
      shape0 = {seq_len, num_dir, batch_size, hidden_size};
    }
    shape1 = {num_dir, batch_size, hidden_size};
    shape2 = {num_dir, batch_size, hidden_size};
  }
  if (module::isNone(getY()) == false) {
    module::setShapeOrVerify(getY(), shape0);
  }
  if (module::isNone(getYH()) == false) {
    module::setShapeOrVerify(getYH(), shape1);
  }
  if (module::isNone(getYC()) == false) {
    module::setShapeOrVerify(getYC(), shape2);
  }
}
void ops::LSTMOp::type_inference(){
  if(!module::isNone(getY()))
    module::setElementType(getY(), module::getElementType(getInput()));
  if(!module::isNone(getYH()))
    module::setElementType(getYH(), module::getElementType(getInput()));
  if(!module::isNone(getYC()))
    module::setElementType(getYC(), module::getElementType(getInput()));
}
