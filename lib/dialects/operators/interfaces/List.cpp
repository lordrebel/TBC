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
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
/*
auto ins = op->getOperands();
  auto outs = op->getResults();
  auto num_in = ins.size();
  auto num_out = outs.size();
  std::vector<float> datas[num_out];
  for (int i = 0; i < num_out; i++) {
    if (module::isNone(outs[i])) {
      continue;
    }
    auto num_elem = module::getNumElements(outs[i]);
    datas[i].assign(num_elem, 0.0f);
  }
  std::vector<std::shared_ptr<std::vector<float>>> inputs;
  for (int i = 0; i < num_in; i++) {
    if (false == module::isWeight(ins[i])) {
      inputs.push_back(nullptr);
      continue;
    }
    auto in_op = cast<top::WeightOp>(ins[i].getDefiningOp());
    auto d = in_op.read<float>();
    inputs.push_back(d);
  }
  InferenceParameter p;
  for (int i = 0; i < num_in; i++) {
    if (inputs[i] == nullptr) {
      p.inputs.push_back(nullptr);
    } else {
      p.inputs.push_back(inputs[i]->data());
    }
  }
  for (int i = 0; i < num_out; i++) {
    p.outputs.push_back(datas[i].data());
  }
  auto ret = infer.init(p);
  assert(mlir::succeeded(ret));
  ret = infer.inference(p);
  assert(mlir::succeeded(ret));
  infer.deinit(p);
  OpBuilder builder(module::getCtx());
  builder.setInsertionPointAfter(op);
  for (int i = 0; i < num_out; i++) {
    if (datas[i].empty()) {
      continue;
    }
    std::string suffix = std::string("folder_") + std::to_string(i);
    auto out = outs[i];
    auto out_type = out.getType().cast<RankedTensorType>();
    auto new_type =
RankedTensorType::get(out_type.getShape(),builder.getF32Type()); auto new_op =
top::WeightOp::create(op, "folder", datas[i], new_type);
    out.replaceAllUsesWith(new_op);
  }
LogicalResult top::ListOp::inference(InferenceParameter &p) {
  int64_t offset = 0;
  int64_t num_inputs = getInputs().size();
  for (int i = 0; i < num_inputs; i++) {
    if (module::isNone(getInputs()[i])) {
      continue;
    }
    auto num = module::getNumElements(getInputs()[i]);
    memcpy(p.outputs[0] + offset, p.inputs[i], num * sizeof(float));
    offset += num;
  }
  return success();
}
*/
// ListOp is special, will convert to WeightOp
void ops::ListOp::shape_inference() {

  int64_t num_outputs = 0;
  for (auto in : getInputs()) {
    if (module::isNone(in)) {
      continue;
    }
    num_outputs += module::getNumElements(in);
  }
  std::vector<int64_t> new_shape = {num_outputs};
  module::setShapeOrVerify(getOutput(), new_shape);

  if (module::isAllWeight(getOperation())) {
    // fold to weight
    std::vector<std::vector<float>> inputs;
    std::vector<int64_t> lengths;
    for (auto in : getInputs()) {
      if (module::isNone(in)) {
        continue;
      }

      auto weightop = llvm::cast<WeightOp>(in.getDefiningOp());
      auto data = weightop.read_as_float();
      inputs.push_back(*data);
      lengths.push_back(data->size());
    }
    std::vector<float> output(module::getNumElements(getOutput()), 0.0f);
    int64_t offset = 0;
    int64_t num_inputs = inputs.size();
    for (int i = 0; i < num_inputs; i++) {
      auto num = lengths[i];
      memcpy(output.data() + offset, inputs[i].data(), num * sizeof(float));
      offset += num;
    }
    OpBuilder builder(module::getCtx());
    builder.setInsertionPointAfter(getOperation());
    auto out_type = cast<RankedTensorType>(getOutput().getType());
    auto new_type =
        RankedTensorType::get(out_type.getShape(), builder.getF32Type());
    auto new_op =
        ops::WeightOp::create(getOperation(), "folder", output, new_type);
    getOutput().replaceAllUsesWith(new_op);
  }
}

void ops::ListOp::type_inference() {
  llvm_unreachable("not support type inference for list op yet. ");
}
