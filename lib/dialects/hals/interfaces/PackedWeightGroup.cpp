#include "dialects/hals/IR/hals.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "support/module.h"
#include "support/tensorfile.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>
namespace tbc::hals {

template <typename DataT>
llvm::LogicalResult addTensor(TensorFile &t_file, llvm::StringRef name,
                              const std::shared_ptr<std::vector<DataT>> &data,
                              std::vector<int64_t> &shape) {
  const DataT *data_ptr = data->data();
  return t_file.addTensor(name, data_ptr, shape);
}
PackedWeightGroupOp PackedWeightGroupOp::Merge(
    std::vector<PackedWeightGroupOp> &packedweight_groups) {
  std::vector<mlir::Value> weightValues;
  std::vector<mlir::Type> weightTy;
  std::vector<mlir::Location> all_locs;
  std::vector<mlir::Value> originOuts;
  for (auto &group : packedweight_groups) {
    std::copy(group.getOutputs().begin(), group.getOutputs().end(),
              std::back_inserter(originOuts));
    for (auto val : group.getBody().begin()->getTerminator()->getOperands()) {
      weightValues.push_back(val);
      weightTy.push_back(val.getType());
    }
    all_locs.push_back(group->getLoc());
  }
  if (originOuts.size() != weightValues.size()) {
    LOGE << "error: origin outs size not equal to weight values size\n";
    llvm_unreachable(" origin outs size not equal to weight values size");
  }
  OpBuilder builder(packedweight_groups[0]->getContext());
  builder.setInsertionPointToStart(packedweight_groups[0]->getBlock());
  auto final_loc = builder.getFusedLoc(all_locs);
  auto group_op =
      builder.create<hals::PackedWeightGroupOp>(final_loc, weightTy);
  auto block = builder.createBlock(&(group_op.getBody()));
  builder.setInsertionPointToStart(block);
  auto groupReturnOp = builder.create<hals::ReturnOp>(
      mlir::UnknownLoc::get(group_op->getContext()), weightValues);

  for (int i = weightValues.size() - 1; i >= 0; i--) {
    if (i == weightValues.size() - 1) {
      weightValues[i].getDefiningOp()->moveBefore(groupReturnOp);
    } else {
      weightValues[i].getDefiningOp()->moveBefore(
          weightValues[i + 1].getDefiningOp());
    }
  }

  for (size_t i = 0; i != originOuts.size(); i++) {
    originOuts[i].replaceAllUsesWith(group_op.getOutputs()[i]);
  }

  for (auto &op : packedweight_groups) {
    op->erase();
  }

  return group_op;
}

std::vector<int64_t> PackedWeightGroupOp::getOffsets() {
  std::vector<int64_t> offsets = {};
  int64_t offset = 0;
  this->getBody().walk([&](hals::WeightOp op) {
    offsets.push_back(offset);
    offset += op.getByteSize();
  });
  return offsets;
}

std::vector<std::pair<mlir::Value, mlir::Value>>
PackedWeightGroupOp::getReturnValueMap() {
  auto outs = this->getOutputs();
  auto return_inputs = mlir::cast<hals::ReturnOp>(
                           this->getBody().getBlocks().begin()->getTerminator())
                           ->getOperands();
  std::vector<std::pair<mlir::Value, mlir::Value>> res;
  for (auto [idx, val] : llvm::enumerate(outs)) {
    res.push_back(std::make_pair(val, return_inputs[idx]));
  }
  return res;
}
llvm::LogicalResult
PackedWeightGroupOp::ToNpzFile(const std::string &filename) {
  TensorFile t_file(filename, false);
  this->getBody().walk([&](hals::WeightOp op) {
    auto val = op.getOutput();
    auto shape = module::getShape(val);
    auto dtype = module::getElementType(val);
    auto v_shape = std::vector<int64_t>(shape.begin(), shape.end());
    if (dtype.isF32()) {
      auto status =
          addTensor(t_file, module::getName(val), op.read<float>(), v_shape);
      if (status.failed()) {
        LOGE << "Failed to add tensor:" << val << " to npz file: " << filename
             << "\n";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    } else if (dtype.isF16()) {
      auto status =
          addTensor(t_file, module::getName(val), op.read<uint16_t>(), v_shape);
      if (status.failed()) {
        LOGE << "Failed to add tensor:" << val << " to npz file: " << filename
             << "\n";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    } else if (dtype.isBF16()) {
      return WalkResult::advance();
    } else if (dtype.isBF16()) {
      auto status =
          addTensor(t_file, module::getName(val), op.read<uint16_t>(), v_shape);
      if (status.failed()) {
        LOGE << "Failed to add tensor:" << val << " to npz file: " << filename
             << "\n";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    } else if (dtype.isInteger(32)) {
      auto status =
          addTensor(t_file, module::getName(val), op.read<int32_t>(), v_shape);
      if (status.failed()) {
        LOGE << "Failed to add tensor:" << val << " to npz file: " << filename
             << "\n";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    } else if (dtype.isInteger(16)) {
      auto status =
          addTensor(t_file, module::getName(val), op.read<int16_t>(), v_shape);
      if (status.failed()) {
        LOGE << "Failed to add tensor:" << val << " to npz file: " << filename
             << "\n";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    } else if (dtype.isInteger(8)) {
      auto status =
          addTensor(t_file, module::getName(val), op.read<int8_t>(), v_shape);
      if (status.failed()) {
        LOGE << "Failed to add tensor:" << val << " to npz file: " << filename
             << "\n";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    } else if (dtype.isUnsignedInteger(8)) {
      auto status =
          addTensor(t_file, module::getName(val), op.read<uint8_t>(), v_shape);
      if (status.failed()) {
        LOGE << "Failed to add tensor:" << val << " to npz file: " << filename
             << "\n";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    } else if (dtype.isFloat8E4M3() || dtype.isFloat8E5M2()) {

      auto status =
          addTensor(t_file, module::getName(val), op.read<uint8_t>(), v_shape);
      if (status.failed()) {
        LOGE << "Failed to add tensor:" << val << " to npz file: " << filename
             << "\n";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    } else {
      LOGE << "Unsupported data type: " << dtype << "\n";
      return WalkResult::interrupt();
    }
  });
  t_file.save(filename);
  return llvm::success();
}
llvm::LogicalResult
PackedWeightGroupOp::ToBinFile(const std::string &filename) {
  FILE *fp = fopen(filename.c_str(), "wb");
  if (!fp) {
    LOGE << "Unable to open file: " << filename << "\n";
    return llvm::failure();
  }
  auto result = this->getBody().walk([&](hals::WeightOp op) -> WalkResult {
    auto val = op.getOutput();
    auto nums = module::getNumElements(val);
    auto dtype = module::getElementType(val);
    if (dtype.isF32()) {
      auto data = op.read<float>();
      fwrite(data->data(), sizeof(float), nums, fp);
      return WalkResult::advance();
    } else if (dtype.isF16()) {
      auto data = op.read<uint16_t>();
      fwrite(data->data(), sizeof(uint16_t), nums, fp);
      return WalkResult::advance();
    } else if (dtype.isBF16()) {
      auto data = op.read<uint16_t>();
      fwrite(data->data(), sizeof(uint16_t), nums, fp);
      return WalkResult::advance();
    } else if (dtype.isInteger(32)) {
      auto data = op.read<int32_t>();
      fwrite(data->data(), sizeof(int32_t), nums, fp);
      return WalkResult::advance();
    } else if (dtype.isInteger(16)) {
      auto data = op.read<int16_t>();
      fwrite(data->data(), sizeof(int16_t), nums, fp);
      return WalkResult::advance();
    } else if (dtype.isInteger(8)) {
      auto data = op.read<int8_t>();
      fwrite(data->data(), sizeof(int8_t), nums, fp);
      return WalkResult::advance();
    } else if (dtype.isUnsignedInteger(8)) {
      auto data = op.read<uint8_t>();
      fwrite(data->data(), sizeof(uint8_t), nums, fp);
      return WalkResult::advance();
    } else if (dtype.isFloat8E4M3() || dtype.isFloat8E5M2()) {
      auto data = op.read<uint8_t>();
      fwrite(data->data(), sizeof(uint8_t), nums, fp);
      return WalkResult::advance();
    } else {
      LOGE << "Unsupported data type: " << dtype << "\n";
      return WalkResult::interrupt();
    }
  });
  fclose(fp);
  if (result.wasInterrupted()) {
    LOGE << "Walk was interrupted, file may be incomplete: " << filename
         << "\n";
    return llvm::failure();
  }
  return llvm::success();
}
} // namespace tbc::hals
