#include "dialects/kernels/IR/Kernels.h"
#include "support/mathutil.h"
#include "support/module.h"
#include "support/targets/params.h"
using namespace tbc::tgt;
template <typename T> static int remove_value(std::vector<T> &v, T value, bool is_int8) {
  int idx = 0;
  for (auto iter = v.begin(); iter != v.end(); iter++, idx++) {
    if (idx == 0 && is_int8)
      continue;
    if (*iter == value) {
      v.erase(iter);
      return idx;
    }
  }
  return -1;
}

slice_attr_t tbc::kls::SliceOp::parseParam() {
  slice_attr_t attr;
  std::vector<int64_t> is = module::getShape(getInput());
  std::vector<int64_t> os = module::getShape(getOutput());
  int num_dims = is.size();
  auto crop_offset = module::getI64Array(getOffset());
  auto crop_steps = module::getI64Array(getSteps());

  assert(crop_offset->size() == crop_steps->size());
  assert(is.size() == crop_steps->size());
  if (is.size() > os.size()) {
    for (int out_dims = os.size(); out_dims < num_dims; out_dims++) {
      os.insert(os.begin(), 1);
    }
  }
  auto input_dtype = getDataType(getInput());
  bool is_int8 = (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8);
  if (num_dims > 4) {
    // remove dims = 1
    while (num_dims > 4) {
      int idx = remove_value<int64_t>(is, 1, is_int8);
      if (idx < 0) {
        break;
      }
      crop_offset->erase(crop_offset->begin() + idx);
      crop_steps->erase(crop_steps->begin() + idx);
      os.erase(os.begin() + idx);
      num_dims--;
    }
    // remove continous
    while (num_dims > 4) {
      bool done = false;
      for (int i = 0; i < num_dims - 1; i++) {
        if (i == 0 && is_int8)
          continue;
        if (is[i] == os[i] && is[i + 1] == os[i + 1]) {
          is[i] *= is[i + 1];
          os[i] *= os[i + 1];
          is.erase(is.begin() + i + 1);
          os.erase(os.begin() + i + 1);
          crop_steps->erase(crop_steps->begin() + i + 1);
          crop_offset->erase(crop_offset->begin() + i + 1);
          num_dims--;
          done = true;
          break;
        }
      }
      if (done == false) {
        break;
      }
    }
    if (num_dims > 4) {
      llvm_unreachable("permute shape not support");
    }
  }
  attr.is_4 = {1, 1, 1, 1};
  attr.os_4 = {1, 1, 1, 1};
  attr.step_4 = {1, 1, 1, 1};
  attr.offset_4 = {0, 0, 0, 0};
  std::vector<int> real_axes;
  attr.no_step = true;
  for (int idx = 0; idx < num_dims; idx++) {
    attr.is_4[idx] = is[idx];
    attr.os_4[idx] = os[idx];
    attr.step_4[idx] = crop_steps->at(idx);
    attr.offset_4[idx] = crop_offset->at(idx);
    if (attr.no_step && crop_steps->at(idx) != 1) {
      attr.no_step = false;
    }
    if (attr.is_4[idx] != attr.os_4[idx]) {
      real_axes.push_back(idx);
    }
  }
  attr.fusible = false;
  return attr;
}
