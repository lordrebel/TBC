//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/float16.h"
#include "support/float8.h"
#include "support/module.h"
#include "support/mathutil.h"
#include <cstdint>

using namespace tbc::ops;

template <typename T> std::shared_ptr<std::vector<T>> WeightOp::read() {
  auto op = getOperation();
  auto type = cast<RankedTensorType>(getOutput().getType());

  return module::weightFile().readTensor<T>(module::getName(op).str(), type);
}

std::shared_ptr<std::vector<float>> WeightOp::read_as_float() {
  auto dtype = module::getStorageType(getOutput());
  if (dtype.isUnsignedInteger(8)) {
    auto data_u8 = read<uint8_t>();
    return std::make_shared<std::vector<float>>(data_u8->begin(),
                                                data_u8->end());
  } else if (dtype.isInteger(8)) {
    auto data_i8 = read<int8_t>();
    return std::make_shared<std::vector<float>>(data_i8->begin(),
                                                data_i8->end());
  } else if (dtype.isF32()) {
    return read<float>();
  } else if (dtype.isF16()) {
    auto data_u16 = read<uint16_t>();
    auto data_f32 = std::make_shared<std::vector<float>>(data_u16->size());
    for (uint64_t i = 0; i < data_u16->size(); i++) {
      data_f32->data()[i] = f16_to_f32(data_u16->data()[i]);
    }
    return data_f32;
  } else if (dtype.isBF16()) {
    auto data_u16 = read<uint16_t>();
    auto data_f32 = std::make_shared<std::vector<float>>(data_u16->size());
    for (uint64_t i = 0; i < data_u16->size(); i++) {
      data_f32->data()[i] = bf16_to_f32(data_u16->data()[i]);
    }
    return data_f32;
  } else if (dtype.isFloat8E4M3FN()) {
    auto data_u8 = read<uint8_t>();
    auto data_f32 = std::make_shared<std::vector<float>>(data_u8->size());
    for (uint64_t i = 0; i < data_u8->size(); i++) {
      data_f32->data()[i] = f8e4m3_to_f32(data_u8->data()[i]);
    }
    return data_f32;
  } else if (dtype.isFloat8E5M2()) {
    auto data_u8 = read<uint8_t>();
    auto data_f32 = std::make_shared<std::vector<float>>(data_u8->size());
    for (uint64_t i = 0; i < data_u8->size(); i++) {
      data_f32->data()[i] = f8e5m2_to_f32(data_u8->data()[i]);
    }
    return data_f32;
  } else if (dtype.isUnsignedInteger(16)) {
    auto data_u16 = read<uint16_t>();
    return std::make_shared<std::vector<float>>(data_u16->begin(),
                                                data_u16->end());
  } else if (dtype.isInteger(16)) {
    auto data_i16 = read<int16_t>();
    return std::make_shared<std::vector<float>>(data_i16->begin(),
                                                data_i16->end());
  } else if (dtype.isUnsignedInteger(32)) {
    auto data_u32 = read<uint32_t>();
    return std::make_shared<std::vector<float>>(data_u32->begin(),
                                                data_u32->end());
  } else if (dtype.isInteger(32)) {
    auto data_i32 = read<int32_t>();
    return std::make_shared<std::vector<float>>(data_i32->begin(),
                                                data_i32->end());
  }else if(dtype.isInteger(64)){
     auto data_i64 = read<int64_t>();
    return std::make_shared<std::vector<float>>(data_i64->begin(),
                                                data_i64->end());
  }
  dump();
  llvm_unreachable("weight data not support read as float now");
  return nullptr;
}

std::shared_ptr<std::vector<int32_t>> WeightOp::read_as_int32() {
  auto dtype = module::getStorageType(getOutput());
  if (dtype.isInteger(32)) {
    return read<int32_t>();
  } else if (dtype.isUnsignedInteger(16)) {
    auto data_u16 = read<uint16_t>();
    return std::make_shared<std::vector<int32_t>>(data_u16->begin(),
                                                  data_u16->end());
  } else if (dtype.isInteger(16)) {
    auto data_i16 = read<int16_t>();
    return std::make_shared<std::vector<int32_t>>(data_i16->begin(),
                                                  data_i16->end());
  } else if (dtype.isUnsignedInteger(8)) {
    auto data_u8 = read<uint8_t>();
    return std::make_shared<std::vector<int32_t>>(data_u8->begin(),
                                                  data_u8->end());
  } else if (dtype.isInteger(8)) {
    auto data_i8 = read<int8_t>();
    return std::make_shared<std::vector<int32_t>>(data_i8->begin(),
                                                  data_i8->end());
  }
  dump();
  llvm_unreachable("weight data not support read as int32 now");
  return nullptr;
}

std::shared_ptr<std::vector<uint8_t>> WeightOp::read_as_byte() {
  auto dtype = module::getStorageType(getOutput());
  if (dtype.isInteger(8) || dtype.isInteger(4)) {
    return read<uint8_t>();
  } else if (dtype.isF32()) {
    auto data_f32 = read<float>();
    auto bytes = data_f32->size() * sizeof(float);
    auto data_u8 = std::make_shared<std::vector<uint8_t>>(bytes);
    memcpy(data_u8->data(), data_f32->data(), bytes);
    return std::move(data_u8);
  } else if (dtype.isInteger(16)) {
    auto data_i16 = read<int16_t>();
    auto bytes = data_i16->size() * sizeof(int16_t);
    auto data_u8 = std::make_shared<std::vector<uint8_t>>(bytes);
    memcpy(data_u8->data(), data_i16->data(), bytes);
    return std::move(data_u8);
  } else if (dtype.isInteger(32)) {
    auto data_i32 = read<int32_t>();
    auto bytes = data_i32->size() * sizeof(int32_t);
    auto data_u8 = std::make_shared<std::vector<uint8_t>>(bytes);
    memcpy(data_u8->data(), data_i32->data(), bytes);
    return std::move(data_u8);
  } else if (isa<Float16Type, BFloat16Type>(dtype)) {
    auto data_u16 = read<uint16_t>();
    auto bytes = data_u16->size() * sizeof(uint16_t);
    auto data_u8 = std::make_shared<std::vector<uint8_t>>(bytes);
    memcpy(data_u8->data(), data_u16->data(), bytes);
    return std::move(data_u8);
  } else if (isa<Float8E4M3FNType, Float8E5M2Type>(dtype)) {
    auto data_f8 = read<uint8_t>();
    auto bytes = data_f8->size();
    auto data_u8 = std::make_shared<std::vector<uint8_t>>(bytes);
    memcpy(data_u8->data(), data_u8->data(), bytes);
    return std::move(data_u8);
  }
  dump();
  llvm_unreachable("weight data not support read now");
  return nullptr;
}

template <typename T>
Value WeightOp::create(Operation *OwnerOp, llvm::StringRef suffix,
                       const std::vector<T> &data, RankedTensorType &type) {
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  std::string op_name = module::getName(OwnerOp).str();
  std::string new_name = op_name + "_" + suffix.str();
  std::set<StringRef> all_tensor_names;
  module::weightFile().getAllNames(all_tensor_names);
  auto it = all_tensor_names.find(new_name.c_str());
  int index = 1;
  while (it != all_tensor_names.end()) {
    new_name = op_name + "_" + std::to_string((index++)) + "_" + suffix.str();
    it = all_tensor_names.find(new_name.c_str());
  }
  auto ret = module::weightFile().addTensor(new_name, &data, type);
  assert(succeeded(ret));
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp =
      builder.create<WeightOp>(NameLoc::get(nameAttr), type, ValueRange{});

  return newOp.getResult();
}

template std::shared_ptr<std::vector<float>> WeightOp::read();
template std::shared_ptr<std::vector<int8_t>> WeightOp::read();
template std::shared_ptr<std::vector<int16_t>> WeightOp::read();
template std::shared_ptr<std::vector<uint16_t>> WeightOp::read();
template std::shared_ptr<std::vector<uint8_t>> WeightOp::read();
template i32_array_t WeightOp::read();
template i64_array_t WeightOp::read();
template f64_array_t WeightOp::read();
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<float> &data,
                                RankedTensorType &type);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<int16_t> &data,
                                RankedTensorType &type);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<uint16_t> &data,
                                RankedTensorType &type);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<int8_t> &data,
                                RankedTensorType &type);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<uint8_t> &data,
                                RankedTensorType &type);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<int32_t> &data,
                                RankedTensorType &type);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<uint32_t> &data,
                                RankedTensorType &type);
template Value WeightOp::create(Operation *OwnerOp, llvm::StringRef name,
                                const std::vector<int64_t> &data,
                                RankedTensorType &type);

Value WeightOp::clone_bf16(Operation *OwnerOp, std::string name) {
  auto type = cast<RankedTensorType>(getType());
  auto dtype = type.getElementType();
  assert(dtype.isF32());
  auto data = read<float>();
  auto count = data->size();
  auto data_bf16 = std::make_shared<std::vector<uint16_t>>(count);

#pragma omp parallel for schedule(static, omp_schedule(count))
  for (uint32_t i = 0; i < count; i++) {
    data_bf16->at(i) = f32_to_bf16(data->at(i));
  }
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  // if the weightop will be used by 2 ops, it need to create a new WeightOp
  std::string new_name = name.empty() ? module::getName(OwnerOp).str() +
                         module::getName(getOperation()).str() + "_bf16" : name;
  auto new_type = RankedTensorType::get(type.getShape(), builder.getBF16Type());
  auto ret =
      module::weightFile().addTensor(new_name, data_bf16->data(), new_type);
  assert(succeeded(ret));
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<WeightOp>(NameLoc::get(nameAttr), new_type,
                                             ValueRange{});
  return newOp.getResult();
};

Value WeightOp::clone_f16(Operation *OwnerOp) {
  auto type = cast<RankedTensorType>(getType());
  auto dtype = type.getElementType();
  assert(dtype.isF32());
  auto data = read<float>();
  auto count = data->size();
  auto data_f16 = std::make_shared<std::vector<uint16_t>>(count);

#pragma omp parallel for schedule(static, omp_schedule(count))
  for (uint32_t i = 0; i < count; i++) {
    data_f16->at(i) = f32_to_f16(data->at(i));
  }
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  // if the weightop will be used by 2 ops, it need to create a new WeightOp
  std::string new_name = module::getName(OwnerOp).str() +
                         module::getName(getOperation()).str() + "_f16";
  auto new_type = RankedTensorType::get(type.getShape(), builder.getF16Type());
  auto ret =
      module::weightFile().addTensor(new_name, data_f16->data(), new_type);
  assert(succeeded(ret));
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<WeightOp>(NameLoc::get(nameAttr), new_type,
                                             ValueRange{});
  return newOp.getResult();
};

Value WeightOp::clone_f8e4m3(Operation *OwnerOp, bool per_channel_scale) {
  auto type = cast<RankedTensorType>(getType());
  auto shape = type.getShape();
  auto dtype = type.getElementType();
  assert(dtype.isF32());
  auto data = read<float>();
  auto count = data->size();
  auto cnt_p_c = count / shape[0];
  auto data_f8 = std::make_shared<std::vector<uint8_t>>(count);

  f64_array_t weight_scale_v;
  if (per_channel_scale) {
    if (getScale().has_value()) {
      weight_scale_v = module::getF64Array(getScale().value());
      assert(shape[0] == weight_scale_v->size());
    }
    else {
      // search for the max value and set scale to it
      std::vector<double> weight_scale_v_;
      for (size_t i=0;i<shape[0];i++) {
        float absmax = std::abs(data->at(i*cnt_p_c));
        for (size_t j=0;j<cnt_p_c;j++) {
          absmax = std::abs(data->at(i*cnt_p_c+j)) > absmax ? std::abs(data->at(i*cnt_p_c+j)) : absmax;
        }
        absmax = absmax > 1e-8 ? absmax: 1e-8;
        weight_scale_v_.push_back(absmax / get_f8e4m3_max());
      }
      weight_scale_v = std::make_shared<std::vector<double>>(weight_scale_v_);
    }
#pragma omp parallel for schedule(static, omp_schedule(count))
    for (uint32_t i = 0; i < count; i++) {
      data->at(i) = data->at(i)/weight_scale_v.get()->at((int)(i/cnt_p_c));
    }
  } else {
    float absmax = std::abs(data->at(0));
    for (int i=0;i<count;i++)
      absmax = absmax>std::abs(data->at(i)) ? absmax : std::abs(data->at(i));
    weight_scale_v = std::make_shared<std::vector<double>>(1, absmax / get_f8e4m3_max());
#pragma omp parallel for schedule(static, omp_schedule(count))
    for (uint32_t i = 0; i < count; i++) {
      data->at(i) = data->at(i)/weight_scale_v.get()->at(0);
    }
  }
#pragma omp parallel for schedule(static, omp_schedule(count))
  for (uint32_t i = 0; i < count; i++) {
    data_f8->at(i) = f32_to_f8e4m3(data->at(i));
  }
  // FIXME: should calculate the scale and set the scale attr
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);

  builder.setInsertionPoint(OwnerOp);
  // if the weightop will be used by 2 ops, it need to create a new WeightOp
  std::string new_name = module::getName(OwnerOp).str() + module::getName(getOperation()).str() + "_f8e4m3";
  auto new_type = RankedTensorType::get(type.getShape(),builder.getFloat8E4M3FNType()); // builder.getFloat8E5M2Type());
  auto ret =
      module::weightFile().addTensor(new_name, data_f8->data(), new_type);
  assert(succeeded(ret));
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<WeightOp>(NameLoc::get(nameAttr), new_type,
                                             ValueRange{});
  if (!getScale().has_value()) {
    newOp.getOperation()->setAttr("scale", builder.getF64ArrayAttr(ArrayRef<double>{*weight_scale_v}));
  }

  return newOp.getResult();
};

Value WeightOp::clone_f8e5m2(Operation *OwnerOp) {
  auto type = cast<RankedTensorType>(getType());
  auto dtype = type.getElementType();
  assert(dtype.isF32());
  auto data = read<float>();
  auto count = data->size();
  auto data_f8 = std::make_shared<std::vector<uint8_t>>(count);

#pragma omp parallel for schedule(static, omp_schedule(count))
  for (uint32_t i = 0; i < count; i++) {
    data_f8->at(i) = f32_to_f8e5m2(data->at(i));
  }
  // FIXME: scale set to 1.0
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  // if the weightop will be used by 2 ops, it need to create a new WeightOp
  std::string new_name = module::getName(OwnerOp).str() +
                         module::getName(getOperation()).str() + "_f8e5m2";
  auto new_type = RankedTensorType::get(type.getShape(),builder.getFloat8E5M2Type()); // builder.getFloat8E5M2Type());
  auto ret =
      module::weightFile().addTensor(new_name, data_f8->data(), new_type);
  assert(succeeded(ret));
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<WeightOp>(NameLoc::get(nameAttr), new_type,
                                             ValueRange{});
  return newOp.getResult();
};

// template <typename Ty>
Value WeightOp::clone_int(Operation *OwnerOp) {
  auto type = cast<RankedTensorType>(getType());
  auto dtype = type.getElementType();
  assert(dtype.isF32());
  auto data = read<float>();
  auto count = data->size();
  auto data_f16 = std::make_shared<std::vector<int32_t>>(count);

#pragma omp parallel for schedule(static, omp_schedule(count))
  for (uint32_t i = 0; i < count; i++) {
    data_f16->at(i) = static_cast<int32_t>(data->at(i));
  }
  auto ctx = OwnerOp->getContext();
  OpBuilder builder(ctx);
  builder.setInsertionPoint(OwnerOp);
  // if the weightop will be used by 2 ops, it need to create a new WeightOp
  std::string new_name = module::getName(OwnerOp).str() +
                         module::getName(getOperation()).str() + "_int";
  auto new_type = RankedTensorType::get(type.getShape(), builder.getI32Type());
  auto ret =
      module::weightFile().addTensor(new_name, data_f16->data(), new_type);
  assert(succeeded(ret));
  auto nameAttr = builder.getStringAttr(new_name);
  auto newOp = builder.create<WeightOp>(NameLoc::get(nameAttr), new_type,
                                             ValueRange{});
  return newOp.getResult();
};

Value WeightOp::clone(llvm::StringRef suffix) {
  auto ctx = getContext();
  OpBuilder builder(ctx);
  auto op = getOperation();
  builder.setInsertionPointAfter(op);
  auto name = module::getName(op);
  auto new_name = name.str() + "_" + suffix.str();
  auto nameAttr = builder.getStringAttr(new_name);
  auto ret = module::weightFile().cloneTensor(name, suffix);
  assert(succeeded(ret));
  auto newOp = builder.create<WeightOp>(NameLoc::get(nameAttr), getType(),
                                             ValueRange{});
  return newOp.getOutput();
}

Value WeightOp::split(int begin, int end, int axis, mlir::Type to_type, std::string suffix) {
  auto op = getOperation();
  auto shape = module::getShape(getOutput());
  auto dim = shape.size();
  axis = axis < 0 ? dim + axis : axis;
  int64_t outer = 1;
  for (int i = 0; i < axis; ++i) {
    outer *= shape[i];
  }
  int64_t inner = module::getNumElements(getOutput()) / outer;
  int64_t head_inner = inner / shape[axis] * (end - begin);
  auto out_weight = std::make_shared<std::vector<float_t>>(outer * head_inner);
  auto weight_op = read_as_float();
  for (int64_t i = 0; i < outer; ++i) {
    int64_t src_offset = i * inner + begin * (inner / shape[axis]);
    int64_t dst_offset = i * head_inner;
    for (int64_t j = 0; j < head_inner; ++j) {
      out_weight->data()[dst_offset + j] = weight_op->at(src_offset + j);
    }
  }
  std::vector<int64_t> out_shape(shape);
  out_shape[axis] = end - begin;
  auto new_type = RankedTensorType::get(out_shape, to_type);
  return create(op, suffix, *out_weight, new_type);
}

template <typename T>
LogicalResult WeightOp::update(const std::vector<T> &data, size_t count) {
  auto op = getOperation();
  return module::weightFile().updateTensorData(module::getName(op).str(),
                                               &data[0], count);
}

template LogicalResult WeightOp::update(const std::vector<uint8_t> &data,
                                        size_t cont);
template LogicalResult WeightOp::update(const std::vector<uint16_t> &data,
                                        size_t cont);
template LogicalResult WeightOp::update(const std::vector<uint32_t> &data,
                                        size_t cont);
template LogicalResult WeightOp::update(const std::vector<float> &data,
                                        size_t cont);
