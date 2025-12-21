//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "support/module.h"
#include "dialects/operators/IR/operator.h"
#include "dialects/hals/IR/hals.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "support/mathutil.h"
#include "support/tensorfile.h"
#include "support/utils.h"
#include <cstdint>
using namespace mlir;
using namespace mlir::func;
using namespace tbc::utils;
namespace tbc {
namespace module {
struct Attr {
  static constexpr llvm::StringRef COMPILE_PHASE = "module.compile_phase";
  static constexpr llvm::StringRef TARGET = "module.target";
  static constexpr llvm::StringRef WEIGHT_FILE = "module.weight_file";

  static constexpr llvm::StringRef ASYMMETRIC = "module.asymmetric";
  static constexpr llvm::StringRef MODE = "module.mode";
  static constexpr llvm::StringRef PLATFORM = "module.platform";
  static constexpr llvm::StringRef INPUTS = "module.inputs";
  static constexpr llvm::StringRef OUTPUTS = "module.outputs";
  static constexpr llvm::StringRef TRAIN = "module.train";
  static constexpr llvm::StringRef QUANT_GROUP_SIZE = "module.q_group_size";
};

static ModuleOp m = nullptr;
static MLIRContext *ctx = nullptr;
static Target target = Target::ALL;
static Platform platform = Platform::ONNX;
static std::unique_ptr<tbc::TensorFile> wFile = nullptr;
static std::string weightFileName = "";

void init(ModuleOp module) {
  m = module;
  ctx = m.getContext();
  auto target_ = m->getAttrOfType<StringAttr>(Attr::TARGET);
  target = symbolizeTarget(target_).value_or(Target::ALL);
  wFile = nullptr;
  if (m->hasAttrOfType<StringAttr>(Attr::PLATFORM)) {
    auto p = m->getAttrOfType<StringAttr>(Attr::PLATFORM);
    platform = symbolizePlatform(p).value_or(Platform::ONNX);
  } else {
    platform = Platform::ONNX;
  }
}

ops::NoneOp getNoneOp(Operation *op) {
  assert(op != nullptr);
  if (auto noneOp = dyn_cast<ops::NoneOp>(op)) {
    return noneOp;
  }
  FuncOp funcOp;
  if (isa<FuncOp>(op)) {
    funcOp = cast<FuncOp>(op);
  } else {
    funcOp = cast<FuncOp>(op->getParentOp());
  }
  auto &block = funcOp.front();
  auto &topOp = block.front();
  if (auto noneOp = dyn_cast<ops::NoneOp>(topOp)) {
    return noneOp;
  }
  auto ctx = op->getContext();
  auto builder = OpBuilder(ctx);
  builder.setInsertionPointToStart(&block);
  auto NoneOp = builder.create<ops::NoneOp>(builder.getUnknownLoc(),
                                            builder.getNoneType());
  return NoneOp;
}

static ModuleOp getModuleOp(Value v) {
  auto parent_op = v.getParentBlock()->getParentOp();
  while (parent_op != nullptr && !isa<ModuleOp>(parent_op)) {
    parent_op = parent_op->getParentOp();
  }
  if (parent_op == nullptr) {
    return nullptr;
  }
  return cast<ModuleOp>(parent_op);
}

static ModuleOp getModuleOp(Operation *op) {
  while (op != nullptr && !isa<ModuleOp>(op)) {
    op = op->getParentOp();
  }
  if (op == nullptr) {
    return nullptr;
  }
  return cast<ModuleOp>(op);
}

Value getOriValue(Value v) {
  auto s = getModuleOp(v);
  if (!s) {
    return v;
  }
  if (auto block_arg = dyn_cast_or_null<BlockArgument>(v)) {
    int idx = block_arg.getArgNumber();
    // blockargument have multi-layers nest.
    FuncOp func_op;
    if (isa<FuncOp>(v.getParentBlock()->getParentOp()))
      func_op = cast<FuncOp>(v.getParentBlock()->getParentOp());
    else
      func_op = v.getParentBlock()->getParentOp()->getParentOfType<FuncOp>();

    if (func_op) {
      // cur call op
      auto call_op = getCallOp(func_op);
      // pre call op
      auto operand = call_op.getOperand(idx);
      if (isa<BlockArgument>(operand)) {
        auto find_root = [](auto &&Me, Value v) -> Value {
          if (isa<BlockArgument>(v)) {
            int index = dyn_cast<BlockArgument>(v).getArgNumber();
            FuncOp func_op;
            if (isa<FuncOp>(v.getParentBlock()->getParentOp()))
              func_op = cast<FuncOp>(v.getParentBlock()->getParentOp());
            else
              func_op =
                  v.getParentBlock()->getParentOp()->getParentOfType<FuncOp>();
            auto call_op = getCallOp(func_op);
            return Me(Me, call_op.getOperand(index));
          } else {
            return v;
          }
        };

        Value src_v = find_root(find_root, operand);
        return src_v;
      }
      auto result = cast<OpResult>(operand);
      auto opd = result.getDefiningOp();
      if (isa<ops::InputOp>(opd)) {
        return operand;
      }
      auto pre_call_op = dyn_cast<func::CallOp>(opd);
      auto pre_func_op = getFuncOp(s, pre_call_op.getCallee());
      auto return_op = dyn_cast<ReturnOp>(pre_func_op.front().back());
      return return_op.getOperand(result.getResultNumber());
    }
  } else if (auto pre_op = v.getDefiningOp()) {
    if (isa<func::CallOp>(pre_op)) {
      auto call_op = dyn_cast<func::CallOp>(pre_op);
      int index = cast<OpResult>(v).getResultNumber();
      for (auto func : s.getOps<FuncOp>()) {
        if (call_op.getCallee() == func.getName()) {
          Block &entryBlock = func.front();
          auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
          return returnOp->getOperand(index);
        }
      }
    } else {
      return v;
    }
  }

  llvm_unreachable("Failed to get preOperation.FIx me");
}

Operation *getNextOp(Operation *op, int i) {
  Operation *nextOp = nullptr;
  if (op->getResult(i).hasOneUse()) {
    for (auto &use : op->getResult(i).getUses()) {
      nextOp = use.getOwner();
      break;
    }
    assert(nextOp && "nextOp is nullptr");
  } else {
    auto users = op->getUsers();
    if (1 == std::distance(users.begin(), users.end())) {
      nextOp = *users.begin();
    }
  }
  // if not found, will return NULL
  return nextOp;
}

Value getOperand(Operation *op, int i) {
  auto v = op->getOperand(i);
  return getOriValue(v);
}

static void updateModuleTypes(ModuleOp s) {
  Builder builder(ctx);
  // update callee func's return types
  for (auto func : s.getOps<FuncOp>()) {
    if (func.getName() == "main") {
      continue;
    }
    std::vector<Type> returns;
    Block &entryBlock = func.front();
    auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
    for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
      returns.push_back(returnOp->getOperand(i).getType());
    }
    auto fnType = builder.getFunctionType(func.getArgumentTypes(),
                                          llvm::ArrayRef<Type>{returns});
    func.setType(fnType);
    auto callee = getCallOp(func);
    if (callee) {
      for (auto it : llvm::zip(callee.getResults(), returns)) {
        std::get<0>(it).setType(std::get<1>(it));
      }
    }
  }
  // update callee arg types
  for (auto func : s.getOps<FuncOp>()) {
    if (func.getName() == "main") {
      continue;
    }
    auto callee = getCallOp(func);
    if (!callee) {
      continue;
    }
    std::vector<Type> arguments;
    for (auto it :
         llvm::zip(callee.getOperandTypes(), func.front().getArguments())) {
      arguments.push_back(std::get<0>(it));
      std::get<1>(it).setType(std::get<0>(it));
    }
    auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>(arguments),
                                          func.getResultTypes());
    func.setType(fnType);
  }
  // update main op return types
  auto mainFunc = getMainFuncOp(s);
  Block &entryBlock = mainFunc.front();
  auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
  std::vector<Type> returns;
  for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
    returns.push_back(returnOp->getOperand(i).getType());
  }
  std::vector<Type> inputs;
  auto args = mainFunc.getArguments();
  for (auto arg : args) {
    inputs.push_back(arg.getType());
  }
  auto fnType = builder.getFunctionType(llvm::ArrayRef<Type>{inputs},
                                        llvm::ArrayRef<Type>{returns});
  mainFunc.setType(fnType);
}

void updateModuleTypes() {
  auto modules = getAllModules();
  for (auto s : *modules) {
    updateModuleTypes(s);
  }
}

static void removeUnusedOp(ModuleOp submodule) {
  std::vector<Operation *> all_ops;
  for (auto func : submodule.getOps<FuncOp>()) {
    // for to support nested region's op
    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!isa<ReturnOp, FuncOp, ops::YieldOp,hals::ReturnOp>(op))
        all_ops.push_back(op);
    });
  }
  for (auto iter = all_ops.rbegin(); iter != all_ops.rend(); iter++) {
    if ((*iter)->use_empty()) {
      (*iter)->erase();
    }
  }
}

void removeUnusedOp() {
  auto modules = getAllModules();
  for (auto s : *modules) {
    removeUnusedOp(s);
  }
}

size_t getBytes(Value v) {
  if (isa<NoneType>(v.getType())) {
    return 0;
  }
  auto type = cast<RankedTensorType>(v.getType());
  auto elm_count = type.getNumElements();
  auto etype = getStorageType(v);
  int elm_bits = etype.getIntOrFloatBitWidth();
  return align_up(elm_count * elm_bits, (int64_t)8) / 8;
}

double getDtypeSize(Value v) {
  auto etype = getStorageType(v);
  double elm_bytes = (double)etype.getIntOrFloatBitWidth() / 8;
  return elm_bytes;
}

int64_t getNumElements(Value v) {
  if (isa<RankedTensorType>(v.getType()) == false) {
    return 0;
  }
  auto type = cast<RankedTensorType>(v.getType());
  return type.getNumElements();
}

llvm::ArrayRef<int64_t> getShape(Value v) {
  if (isa<NoneType>(v.getType())) {
    v.dump();
    llvm_unreachable("v is none type");
  }
  if (!isUnranked(v)) {
    auto type = cast<RankedTensorType>(v.getType());
    return type.getShape();
  } else {
    return cast<UnrankedTensorType>(v.getType()).getShape();
  }
}

void setShape(Value v, llvm::ArrayRef<int64_t> shape) {
  auto newType = RankedTensorType::get(shape, getElementType(v));
  v.setType(newType);
}

void getContinousStride(int *stride, int *shape) {
  stride[3] = 1;
  stride[2] = shape[3];
  stride[1] = shape[3] * shape[2];
  stride[0] = stride[1] * shape[1];
}

i32_array_t getI32Array(ArrayAttr arrayAttr) {
  auto data = std::make_shared<std::vector<int32_t>>();
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = dyn_cast<IntegerAttr>(en.value());
    if (attr) {
      data->push_back(attr.getInt());
    } else {
      arrayAttr.dump();
      llvm_unreachable("not int32_t type");
    }
  }
  return std::move(data);
}

i32_array_t getI32Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int32_t default_value) {
  if (arrayAttr.has_value()) {
    auto arr = getI32Array(arrayAttr.value());
    assert(arr->size() == num_elem);
    return std::move(arr);
  }
  return std::make_shared<std::vector<int32_t>>(num_elem, default_value);
}

i64_array_t getI64Array(ArrayAttr arrayAttr) {
  auto data = std::make_shared<std::vector<int64_t>>();
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = dyn_cast<IntegerAttr>(en.value());
    if (attr) {
      data->push_back(attr.getInt());
    } else {
      arrayAttr.dump();
      llvm_unreachable("not int64_t type");
    }
  }
  return std::move(data);
}

i64_array_t getI64Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int64_t default_value) {
  if (arrayAttr.has_value()) {
    auto arr = getI64Array(arrayAttr.value());
    assert(arr->size() == num_elem);
    return std::move(arr);
  }
  return std::make_shared<std::vector<int64_t>>(num_elem, default_value);
}

f64_array_t getF64Array(ArrayAttr arrayAttr) {
  auto data = std::make_shared<std::vector<double>>();
  for (auto en : llvm::enumerate(arrayAttr)) {
    auto attr = dyn_cast<FloatAttr>(en.value());
    data->push_back(attr.getValueAsDouble());
  }
  return std::move(data);
}

f64_array_t getF64Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        double default_value) {
  if (arrayAttr.has_value()) {
    auto arr = getF64Array(arrayAttr.value());
    assert(arr->size() == num_elem);
    return std::move(arr);
  }
  return std::make_shared<std::vector<double>>(num_elem, default_value);
}

Type getStorageType(Type type) {
  if (isa<RankedTensorType>(type)) {
    type = cast<RankedTensorType>(type).getElementType();
  }
  return type;
}

Type getStorageType(Value v) { return getStorageType(v.getType()); }

Type getElementType(Value v) {
  auto type = v.getType();
  if (isa<RankedTensorType>(type)) {
    auto rtype = cast<RankedTensorType>(type);
    return rtype.getElementType();
  } else if (isa<UnrankedTensorType>(type)) {
    auto rtype = cast<UnrankedTensorType>(type);
    return rtype.getElementType();
  }
  return type;
}

void setElementType(Value v, Type type) {
  if (isUnranked(v)) {
    v.setType(UnrankedTensorType::get(type));
  } else if (isa<RankedTensorType>(v.getType())) {
    auto shape = getShape(v);
    auto newType = RankedTensorType::get(shape, type);
    v.setType(newType);
  } else if (isa<NoneType>(v.getType())) {
    llvm::outs() << "v is NoneType skip setting element type\n";

  } else {
    v.getDefiningOp()->getParentOp()->dump();
    v.dump();
    llvm_unreachable("setElementType failed");
  }
}
bool allInputsAreNone(mlir::Operation *op) {
  return std::all_of(
      op->getOperands().begin(), op->getOperands().end(),
      [](mlir::Value v) { return isa<mlir::NoneType>(v.getType()); });
}
bool allInputsAreSameElementType(mlir::Operation *op) {
  auto nums = op->getNumOperands();
  if (nums < 1)
    return true;
  Type firstType = nullptr;
  for (size_t i = 0; i < nums; i++) {
    auto type = op->getOperand(i).getType();
    if (isa<mlir::NoneType>(type))
      continue;
    else {
      firstType = module::getElementType(op->getOperand(i));
      break;
    }
  }
  if (firstType == nullptr)
    return true; // 全部是NoneType

  for (size_t i = 0; i < nums; i++) {
    auto type = op->getOperand(i).getType();
    if (mlir::isa<mlir::NoneType>(type))
      continue; // 跳过NoneType
    if (module::getElementType(op->getOperand(i)) != firstType) {
      return false;
    }
  }
  return true;
}
bool allInputsAreFloatElementType(mlir::Operation *op) {
  size_t nums = op->getNumOperands();
  if (nums < 1)
    return false;
  for (size_t i = 0; i < nums; i++) {
    auto type = op->getOperand(i).getType();
    if (mlir::isa<mlir::NoneType>(type))
      continue;
    if (!mlir::isa<FloatType>(getElementType(op->getOperand(i)))) {
      return false;
    }
  }
  return true;
}
bool allInputsAreIntElementType(mlir::Operation *op) {
  size_t nums = op->getNumOperands();
  if (nums < 1)
    return false;
  for (size_t i = 0; i < nums; i++) {
    auto type = op->getOperand(i).getType();
    if (mlir::isa<mlir::NoneType>(type))
      continue;
    if (!mlir::isa<mlir::IntegerType>(getElementType(op->getOperand(i)))) {
      return false;
    }
  }
  return true;
}

RankedTensorType getTypeLike(Value v, llvm::ArrayRef<int64_t> shape) {
  return RankedTensorType::get(shape, getElementType(v));
}

static void getNCHW_align_right(llvm::ArrayRef<int64_t> &shape, int64_t &n,
                                int64_t &c, int64_t &h, int64_t &w) {
  int num_dims = shape.size();
  n = 1, c = 1, h = 1, w = 1;
  if (num_dims > 0) {
    w = shape[num_dims - 1];
  }
  if (num_dims > 1) {
    h = shape[num_dims - 2];
  }
  if (num_dims > 2) {
    c = shape[num_dims - 3];
  }
  if (num_dims > 3) {
    n = shape[num_dims - 4];
  }
  for (int i = 4; i < num_dims; i++) {
    n *= shape[num_dims - i - 1];
  }
}

static void getNCHW_align_left(llvm::ArrayRef<int64_t> shape, int64_t &n,
                               int64_t &c, int64_t &h, int64_t &w) {
  int num_dims = shape.size();
  n = 1, c = 1, h = 1, w = 1;
  if (num_dims > 0) {
    n = shape[0];
  }
  if (num_dims > 1) {
    c = shape[1];
  }
  if (num_dims > 2) {
    h = shape[2];
  }
  for (size_t i = 3; i < num_dims; ++i) {
    w *= shape[i];
  }
}

void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h,
             int64_t &w, bool left_align) {
  if (left_align) {
    getNCHW_align_left(shape, n, c, h, w);
  } else {
    getNCHW_align_right(shape, n, c, h, w);
  }
}

void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
             bool left_align) {
  auto shape = cast<RankedTensorType>(v.getType()).getShape();
  getNCHW(shape, n, c, h, w, left_align);
}

FuncOp getFuncOp(ModuleOp mod, StringRef func_name) {
  for (auto func : mod.getOps<FuncOp>()) {
    if (func.getName() == func_name) {
      return func;
    }
  }
  llvm::errs() << "Can't find FuncOp:" << func_name << "\n";
  llvm_unreachable("Error getFuncOp !!\n");
  return nullptr;
}

func::CallOp getCallOp(FuncOp func) {
  auto parent = func->getParentOp();
  auto s = cast<ModuleOp>(parent);
  func::CallOp call = nullptr;
  for (auto each_func : s.getOps<FuncOp>()) {
    WalkResult result =
        each_func.walk<WalkOrder::PreOrder>([&](func::CallOp op) {
          if (!call && op.getCallee() == func.getName()) {
            call = op;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      break;
  }
  return call;
}

FuncOp getMainFuncOp(ModuleOp module) { return getFuncOp(module, "main"); }

bool isSign(Value v) {
  auto stype = getStorageType(v);
  if (stype.isUnsignedInteger()) {
    return false;
  }
  return true;
}

bool isWeight(Value v) {
  auto op = v.getDefiningOp();
  if (op == nullptr) {
    return false;
  }
  if (isa<ops::WeightOp>(op)) {
    return true;
  }

  return false;
}

bool isDynWeight(Value v) {
  auto op = v.getDefiningOp();
  if (op == nullptr) {
    return false;
  }
  if (op->hasAttr("dynamic_weight")) {
    // use code below to tag dynamic weight op
    // op->setAttr("dynamic_weight", , rewriter.getBoolAttr(true));
    return true;
  }
  return false;
}

bool isShapeRelatedOp(Value v) {
  auto op = v.getDefiningOp();
  if (op == nullptr) {
    return false;
  }
  if (isa<ops::ShapeOp>(op)) {
    return true;
  }
  return false;
}

bool isAllWeight(Operation *op) {
  for (auto in : op->getOperands()) {
    if (isNone(in) || isWeight(in)) {
      continue;
    }
    return false;
  }
  return true;
}

bool isNone(Value v) { return isa<mlir::NoneType>(v.getType()); }

bool isUnranked(Value v) { return isa<mlir::UnrankedTensorType>(v.getType()); }

bool isDynamicShape(Value v) {
  int ret = false;
  auto tensorTy = dyn_cast<RankedTensorType>(v.getType());
  if (tensorTy) {
    for (int64_t dim : tensorTy.getShape()) {
      if (ShapedType::isDynamic(dim) || dim == 0)
        ret = true;
    }
  }
  return ret;
}

void setShapeOrVerify(Value v, llvm::ArrayRef<int64_t> shape) {
  if (isUnranked(v) || isDynamicShape(v)) {
    auto newType = RankedTensorType::get(shape, getElementType(v));
    v.setType(newType);
  } else {
    auto s = getShape(v);
    /* unranked tensor is okay, for example:
       tensor<*xf32>->tensor<1xf32> */
    if ((std::max(s.size(), shape.size()) > 1) && s != shape) {
      v.getDefiningOp()->getParentOp()->dump();
      v.dump();
      llvm_unreachable("Shape Verify failed");
    }
  }
}

Type DatatypeEnumToType(utils::DataType type, MLIRContext *ctx) {
  switch (type) {
  case utils::DataType::F64:
    return FloatType::getF64(ctx);
  case utils::DataType::F32:
    return FloatType::getF32(ctx);
  case utils::DataType::F16:
    return FloatType::getF16(ctx);
  case utils::DataType::BF16:
    return FloatType::getBF16(ctx);
  case utils::DataType::F8E4M3:
    return FloatType::getFloat8E4M3FN(ctx);
  case utils::DataType::F8E5M2:
    return FloatType::getFloat8E5M2(ctx);
  case utils::DataType::TF32:
    return FloatType::getTF32(ctx);
  case utils::DataType::BOOL:
    return IntegerType::get(ctx, 1);
  case utils::DataType::INT4:
    return IntegerType::get(ctx, 4, IntegerType::Signed);
  case utils::DataType::UINT4:
    return IntegerType::get(ctx, 4, IntegerType::Unsigned);
  case utils::DataType::INT8:
    return IntegerType::get(ctx, 8, IntegerType::Signed);
  case utils::DataType::UINT8:
    return IntegerType::get(ctx, 8, IntegerType::Unsigned);
  case utils::DataType::INT16:
    return IntegerType::get(ctx, 16, IntegerType::Signed);
  case utils::DataType::UINT16:
    return IntegerType::get(ctx, 16, IntegerType::Unsigned);
  case utils::DataType::INT32:
    return IntegerType::get(ctx, 32, IntegerType::Signed);
  case utils::DataType::UINT32:
    return IntegerType::get(ctx, 32, IntegerType::Unsigned);
  case utils::DataType::INT64:
    return IntegerType::get(ctx, 64, IntegerType::Signed);
  case utils::DataType::UINT64:
    return IntegerType::get(ctx, 64, IntegerType::Unsigned);
  default:
    llvm_unreachable("Unsupported Data Type");
  }
}

utils::DataType TypeToDatatypeEnum(Type type) {
  if (type.isF64()) {
    return utils::DataType::F64;
  } else if (type.isF32()) {
    return utils::DataType::F32;
  } else if (type.isF16()) {
    return utils::DataType::F16;
  } else if (type.isBF16()) {
    return utils::DataType::BF16;
  } else if (type.isFloat8E4M3FN()) {
    return utils::DataType::F8E4M3;
  } else if (type.isFloat8E5M2()) {
    return utils::DataType::F8E5M2;
  } else if (type.isTF32()) {
    return utils::DataType::TF32;
  } else if (auto intType = dyn_cast<IntegerType>(type)) {
    unsigned width = intType.getWidth();

    // 检查是否明确指定了符号性
    bool isExplicitlyUnsigned = intType.isUnsigned();

    // 对于signless类型，我们默认将其视为有符号类型
    // 这与大多数编译器的默认行为一致

    if (width == 1) {
      return utils::DataType::BOOL;
    } else if (width == 4) {
      return isExplicitlyUnsigned ? utils::DataType::UINT4
                                  : utils::DataType::INT4;
    } else if (width == 8) {
      return isExplicitlyUnsigned ? utils::DataType::UINT8
                                  : utils::DataType::INT8;
    } else if (width == 16) {
      return isExplicitlyUnsigned ? utils::DataType::UINT16
                                  : utils::DataType::INT16;
    } else if (width == 32) {
      return isExplicitlyUnsigned ? utils::DataType::UINT32
                                  : utils::DataType::INT32;
    } else if (width == 64) {
      return isExplicitlyUnsigned ? utils::DataType::UINT64
                                  : utils::DataType::INT64;
    }
  }

  llvm_unreachable("Unsupported Type to DataType conversion");
}

Target getTarget() { return target; }

Mode getMode() {
  if (false == m->hasAttrOfType<StringAttr>(Attr::MODE)) {
    return Mode::F32;
  }
  auto s = m->getAttrOfType<StringAttr>(Attr::MODE);
  return symbolizeMode(s).value_or(Mode::F32);
}

bool isBF16Modes() {
  auto s = m->getAttrOfType<StringAttr>(Attr::MODE);
  auto mode = symbolizeMode(s).value_or(Mode::F32);
  return mode == Mode::BF16 || mode == Mode::W8BF16 || mode == Mode::W4BF16;
}

bool isF16Modes() {
  auto s = m->getAttrOfType<StringAttr>(Attr::MODE);
  auto mode = symbolizeMode(s).value_or(Mode::F32);
  return mode == Mode::F16 || mode == Mode::W8F16 || mode == Mode::W4F16;
}

bool isF8Modes() {
  auto s = m->getAttrOfType<StringAttr>(Attr::MODE);
  auto mode = symbolizeMode(s).value_or(Mode::F32);
  return mode == Mode::F8 || mode == Mode::F8E4M3 || mode == Mode::F8E5M2;
}

void setTarget(Target target_) {
  target = target_;
  auto s = stringifyTarget(target_);
  m->setAttr(Attr::TARGET, StringAttr::get(m.getContext(), s));
}

bool isTarget(Target target_) { return target == target_; }

void setMode(Mode mode) {
  auto s = stringifyMode(mode);
  m->setAttr(Attr::MODE, StringAttr::get(ctx, s));
}

std::shared_ptr<std::vector<ModuleOp>> getAllModules() {
  auto modules = std::make_shared<std::vector<ModuleOp>>();
  auto sub = m.getOps<ModuleOp>();
  if (sub.empty()) {
    modules->push_back(m);
  } else {
    modules->assign(sub.begin(), sub.end());
  }
  return std::move(modules);
}

int getNumSubModule() {
  auto sub = m.getOps<ModuleOp>();
  return std::distance(sub.begin(), sub.end());
}

bool isAsymmetric() {
  if (m->hasAttrOfType<BoolAttr>(Attr::ASYMMETRIC)) {
    return m->getAttrOfType<BoolAttr>(Attr::ASYMMETRIC).getValue();
  }
  return false;
}

void setAsymmetric(bool is_asymmetric) {
  m->setAttr(Attr::ASYMMETRIC, BoolAttr::get(ctx, is_asymmetric));
}

CompilePhase getCompilePhase() {
  auto s = m->getAttrOfType<StringAttr>(Attr::COMPILE_PHASE);
  return symbolizeCompilePhase(s).value_or(CompilePhase::IMPORTED);
}

Platform getPlatform() { return platform; }

bool isPlatform(Platform plt) { return platform == plt; }

void setCompilePhase(CompilePhase phase) {
  auto s = stringifyCompilePhase(phase);
  m->setAttr(Attr::COMPILE_PHASE, StringAttr::get(ctx, s));
}

void setInputs(ArrayRef<StringRef> inputs) {
  m->setAttr(Attr::INPUTS, Builder(ctx).getStrArrayAttr(inputs));
}
bool isNPU_V1() { return target == Target::NPU_V1; }
bool isNPU_V2() { return target == Target::NPU_V1; }
std::shared_ptr<std::vector<StringRef>> getInputs() {
  auto inputs = m->getAttrOfType<ArrayAttr>(Attr::INPUTS);
  auto data = std::make_shared<std::vector<StringRef>>();
  for (auto en : llvm::enumerate(inputs)) {
    auto attr = dyn_cast<StringAttr>(en.value());
    data->push_back(attr.strref());
  }
  return std::move(data);
}

void setOutputs(ArrayRef<StringRef> outputs) {
  m->setAttr(Attr::OUTPUTS, Builder(ctx).getStrArrayAttr(outputs));
}

std::shared_ptr<std::vector<StringRef>> getOutputs() {
  auto outputs = m->getAttrOfType<ArrayAttr>(Attr::OUTPUTS);
  auto data = std::make_shared<std::vector<StringRef>>();
  for (auto en : llvm::enumerate(outputs)) {
    auto attr = dyn_cast<StringAttr>(en.value());
    data->push_back(attr.strref());
  }
  return std::move(data);
}

bool isCompilePhase(CompilePhase phase) { return phase == getCompilePhase(); }

ModuleOp getModuleOp() { return m; }

Location getLoc() { return m.getLoc(); }

MLIRContext *getCtx() { return ctx; }

uint32_t getIdx(Value v) {
  uint32_t idx = 0;
  if (auto r = dyn_cast<OpResult>(v)) {
    idx = r.getResultNumber();
  } else if (auto r = dyn_cast<BlockArgument>(v)) {
    idx = r.getArgNumber();
  } else {
    v.dump();
    llvm_unreachable("Not Implemented");
  }
  return idx;
}

void setLoc(Value v, NameLoc loc) {
  if (isa<NameLoc>(v.getLoc())) {
    v.setLoc(loc);
    return;
  }
  if (auto fuse_loc = dyn_cast<FusedLoc>(v.getLoc())) {
    std::vector<mlir::Location> locs = fuse_loc.getLocations();
    uint32_t idx = getIdx(v);
    locs[idx] = loc;
    auto new_loc = FusedLoc::get(v.getContext(), locs);
    v.setLoc(new_loc);
    return;
  }
  if (auto op = v.getDefiningOp()) {
    auto op_loc = op->getLoc();
    if (isa<NameLoc>(op_loc)) {
      op->setLoc(loc);
      return;
    }
    if (auto fuse_loc = dyn_cast<FusedLoc>(op->getLoc())) {
      std::vector<mlir::Location> locs = fuse_loc.getLocations();
      auto idx = getIdx(v);
      locs[idx] = loc;
      auto new_loc = FusedLoc::get(v.getContext(), locs);
      op->setLoc(new_loc);
      return;
    }
  }
  v.dump();
  llvm_unreachable("Not Implemented");
}

NameLoc getLoc(Value v) {
  if (auto loc = dyn_cast<NameLoc>(v.getLoc())) {
    return loc;
  } else if (auto fuse_loc = dyn_cast<FusedLoc>(v.getLoc())) {
    auto locs = fuse_loc.getLocations();
    uint32_t idx = getIdx(v);
    if (auto name_loc = dyn_cast<NameLoc>(locs[idx])) {
      return name_loc;
    }
  } else if (auto op = v.getDefiningOp()) {
    auto loc = op->getLoc();
    if (auto name_loc = dyn_cast<NameLoc>(loc)) {
      return name_loc;
    }
    if (auto fuse_loc = dyn_cast<FusedLoc>(loc)) {
      uint32_t idx = getIdx(v);
      auto locs = fuse_loc.getLocations();
      if (auto name_loc = dyn_cast<NameLoc>(locs[idx])) {
        return name_loc;
      }
    }
  }
  v.dump();
  llvm_unreachable("Not Implemented");
  return nullptr;
}

NameLoc getLocLike(Operation *op, llvm::StringRef suffix) {
  return getLocLike(op->getResult(0), suffix);
}

NameLoc getLocLike(Value v, llvm::StringRef suffix) {
  auto name = getName(v);
  auto new_name = name.str() + "_" + suffix.str();
  Builder builder(v.getContext());
  return NameLoc::get(builder.getStringAttr(new_name));
}

void setLocSuffix(Operation *op, llvm::StringRef suffix) {
  if (op->getNumResults() > 1) {
    std::vector<Location> locs;
    for (auto r : op->getResults()) {
      auto loc = getLocLike(r, suffix);
      locs.push_back(loc);
    }
    auto new_loc = FusedLoc::get(op->getContext(), locs);
    op->setLoc(new_loc);
  } else {
    auto loc = getLocLike(op->getResult(0), suffix);
    op->setLoc(loc);
  }
}

StringRef getName(Operation *op, int index) {
  if (auto module = dyn_cast<ModuleOp>(op)) {
    return module.getName().value_or("Unknown");
  }
  if (auto loc = dyn_cast<NameLoc>(op->getLoc())) {
    return loc.getName();
  }
  if (auto loc = dyn_cast<FusedLoc>(op->getLoc())) {
    auto locs = loc.getLocations();
    assert(index < locs.size());
    if (auto name_loc = dyn_cast<NameLoc>(locs[index])) {
      return name_loc.getName();
    }
  }
  op->print(llvm::errs(), OpPrintingFlags().useLocalScope().enableDebugInfo());
  llvm::errs() << "\n";
  llvm_unreachable("op has no name location!!!");
  return "";
}

StringRef getName(Value v) { return getLoc(v).getName().strref(); }

void getInputsOutputs(ModuleOp s, std::vector<Value> &inputs,
                      std::vector<Value> &outputs) {
  auto main_func = getMainFuncOp(s);
  main_func.walk([&](ops::InputOp op) { inputs.push_back(op.getOutput()); });
  main_func.walk([&](ReturnOp op) {
    for (auto out : op.getOperands()) {
      auto result = cast<OpResult>(out);
      auto call_op = result.getDefiningOp<func::CallOp>();
      if (call_op) {
        auto func_op = getFuncOp(s, call_op.getCallee());
        auto return_op = dyn_cast<ReturnOp>(func_op.front().back());
        assert(return_op);
        outputs.push_back(return_op.getOperand(result.getResultNumber()));
      } else {
        outputs.push_back(out);
      }
    }
  });
}

bool isSameOp(Operation *op0, Operation *op1) {
  if (op0 == nullptr || op1 == nullptr) {
    return false;
  }
  if (op0->getName() != op1->getName()) {
    return false;
  }
  if (op0->getNumOperands() != op1->getNumOperands()) {
    return false;
  }
  for (auto it : llvm::zip(op0->getOperands(), op1->getOperands())) {
    if (std::get<0>(it) != std::get<1>(it)) {
      return false;
    }
  }
  if (op0->getNumResults() != op1->getNumResults()) {
    return false;
  }
  for (auto it : llvm::zip(op0->getResultTypes(), op1->getResultTypes())) {
    if (std::get<0>(it) != std::get<1>(it)) {
      return false;
    }
  }
  if (false == op0->getAttrs().equals(op1->getAttrs())) {
    return false;
  }
  return true;
}

void getInputsOutputs(func::CallOp call, std::vector<Value> &inputs,
                      std::vector<Value> &outputs) {
  for (auto opd : call.getOperands()) {
    inputs.emplace_back(module::getOriValue(opd));
  }
  auto md = getModuleOp(call);
  auto func = getFuncOp(md, call.getCallee());
  func.walk([&](ReturnOp op) {
    for (auto output : op.getOperands()) {
      outputs.push_back(output);
    }
  });
}

//-----------------------------------------------------------------
// Helper Functions for weight
//-----------------------------------------------------------------
static std::string genWeightFileName(bool &same_name) {
  auto name = getName(m);
  auto phase = getCompilePhase();
  auto target_ = getTarget();
  auto target = stringifyTarget(target_);
  auto old_name = m->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE).getValue();
  std::string file_name = name.lower() + std::string("_") +
                          stringifyCompilePhase(phase).lower() +
                          std::string("_") + target.lower();
  if (!isTarget(Target::ALL)) {
    auto mode = getMode();
    std::string sym = "";
    if (mode == Mode::INT8) {
      sym = isAsymmetric() ? "_asym" : "_sym";
    }
    auto mode_ = stringifyMode(mode);
    file_name += std::string("_") + mode_.lower() + sym;
  }
  auto new_name = file_name + "_weight.npz";
  same_name = (old_name == new_name);
  if (same_name) {
    new_name = file_name + "_weight_fix.npz";
  }
  return new_name;
}

void saveWeight() {
  // check name conflict
  std::set<StringRef> all_names;
  auto modules = getAllModules();
  for (auto s : *modules) {
    for (auto func : s.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        if (dyn_cast<NameLoc>(op->getLoc()) &&
            !isa<func::ReturnOp, func::CallOp, func::FuncOp, ops::InputOp>(
                op)) {
          auto name = module::getName(op);
          // if op have more than two regions, it can have the same op Name
          if (all_names.find(name) != all_names.end()) {
            op->dump();
            llvm_unreachable("op name conflict");
          }
          all_names.insert(name);
        }
      });
    }
  }
  bool same_name = true;
  std::string filename_;
  if (weightFileName == "") {
    filename_ = module::genWeightFileName(same_name);
  } else {
    same_name = false;
    filename_ = weightFileName;
  }
  // weight remove unused in npz
  if (wFile == nullptr) {
    if (!same_name) {
      weightFile().save(filename_);
      m->setAttr(Attr::WEIGHT_FILE, StringAttr::get(ctx, filename_));
    }
    return;
  }
  if (wFile->changed() == false && same_name) {
    return;
  }
  std::set<StringRef> weight_names;
  for (auto s : *modules) {
    for (auto func : s.getOps<FuncOp>()) {
      func.walk([&](Operation *op) {
        if(isa<ops::WeightOp,hals::WeightOp>(op) == false)
          return;
        weight_names.insert(module::getName(op));
      });
    }
  }
  std::set<StringRef> npz_names;
  wFile->getAllNames(npz_names);
  std::set<StringRef> dif_names;
  for (auto name : npz_names) {
    if (weight_names.find(name) == weight_names.end()) {
      dif_names.insert(name);
    }
  }
  for (auto &name : dif_names) {
    wFile->deleteTensor(name);
  }
  if (wFile->changed() == false && same_name) {
    return;
  }
  wFile->save(filename_);
  m->setAttr(Attr::WEIGHT_FILE, StringAttr::get(ctx, filename_));
}

void setWeightFileName(const std::string &name) { weightFileName = name; }
void detachWeightFile() { wFile = nullptr; }

TensorFile &weightFile() {
  if (wFile == nullptr) {
    auto name = m->getAttrOfType<StringAttr>(Attr::WEIGHT_FILE).getValue();
    wFile = std::make_unique<TensorFile>(name, false);
  }
  return *wFile;
}

//-----------------------------------------------------------------
// Helper for shape op inference
//-----------------------------------------------------------------
void ShapeHelper::bindShapeInfo(const Value &v,
                                const std::vector<int64_t> &shape) {
  _shape_info[v] = shape;
}

std::vector<int64_t> ShapeHelper::getShapeInfo(const Value &v) {
  return _shape_info.at(v);
}

bool ShapeHelper::isShape(const Value &v) {
  return _shape_info.find(v) != _shape_info.end();
}

void bindShapeTensorValue(const Value &v, const std::vector<int64_t> &shape) {
  ShapeHelper::getInstance().bindShapeInfo(v, shape);
}

std::vector<int64_t> getShapeTensorValue(const Value &v) {
  return ShapeHelper::getInstance().getShapeInfo(v);
}

bool isShape(const Value &v) { return ShapeHelper::getInstance().isShape(v); }

} // namespace module
} // namespace tbc
