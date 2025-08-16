#pragma once
#include"utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "dialects/operators/IR/operator.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"

#include "tensorfile.h"

using namespace mlir;
using namespace mlir::func;
using namespace tbc;
using namespace tbc::utils;

namespace tbc {


//-----------------------------------------------------------------
// Types
//-----------------------------------------------------------------
typedef std::shared_ptr<std::vector<int32_t>> i32_array_t;
typedef std::shared_ptr<std::vector<int64_t>> i64_array_t;
typedef std::shared_ptr<std::vector<double>> f64_array_t;
namespace module {

// init module by ModuleOp in init pass
void init(ModuleOp module);

//-----------------------------------------------------------------
// Helper for get/set Attributes
//-----------------------------------------------------------------
int64_t getCoreNum();
void setCoreNum(int64_t core_num = 1);
int64_t getDeviceNum();
void setDeviceNum(int64_t device_num = 1);

Target getTarget();
void setTarget(Target chip);
bool isTarget(Target chip);
Mode getMode();
void setMode(Mode mode);
CompilePhase getCompilePhase();
void setCompilePhase(CompilePhase state);
bool isCompilePhase(CompilePhase state);

void setInputs(ArrayRef<StringRef> inputs);
std::shared_ptr<std::vector<StringRef>> getInputs();
void setOutputs(ArrayRef<StringRef> outputs);
std::shared_ptr<std::vector<StringRef>> getOutputs();
bool isBF16Modes();
bool isF16Modes();
bool isF8Modes();

Platform getPlatform();
bool isPlatform(Platform plt);

bool isAsymmetric();
void setAsymmetric(bool is_asymmetric);

//-----------------------------------------------------------------
// Helper Functions for ModuleOp
//-----------------------------------------------------------------

ModuleOp getModuleOp();
Location getLoc();
MLIRContext *getCtx();

ops::NoneOp getNoneOp(Operation *op);
Value getOriValue(Value v);
Operation *getNextOp(Operation *op, int i = 0);
Value getOperand(Operation *op, int i);
bool isSameOp(Operation *op0, Operation *op1);
void updateModuleTypes();
void removeUnusedOp();
void getNCHW(Value v, int64_t &n, int64_t &c, int64_t &h, int64_t &w,
             bool left_align = true);
void getNCHW(llvm::ArrayRef<int64_t> shape, int64_t &n, int64_t &c, int64_t &h,
             int64_t &w, bool left_align = true);
void getNCDHW(Value v, int64_t &n, int64_t &c, int64_t &d, int64_t &h,
              int64_t &w);
double getDtypeSize(Value v);
size_t getBytes(Value v);
int64_t getNumElements(Value v);
Type getStorageType(Value v); // storage type
Type getStorageType(Type type);
Type getElementType(Value v);
void setElementType(Value v,Type type);
bool allInputsAreSameElementType(mlir::Operation *op);
bool allInputsAreFloatElementType(mlir::Operation *op);
bool allInputsAreIntElementType(mlir::Operation *op);
bool allInputsAreNone(mlir::Operation *op);
RankedTensorType getTypeLike(Value v, llvm::ArrayRef<int64_t> shape);

void setShape(Value v, llvm::ArrayRef<int64_t> shape);
llvm::ArrayRef<int64_t> getShape(Value v);

void getContinousStride(int *stride, int *shape);
bool isUnranked(Value v);
void setShapeOrVerify(Value v, llvm::ArrayRef<int64_t> shape);
bool isSign(Value v);
bool isWeight(Value v);

bool isAllWeight(Operation *op);
bool isNone(Value v);
Type DatatypeEnumToType(utils::DataType type, MLIRContext *ctx);
utils::DataType TypeToDatatypeEnum(Type type);

FuncOp getMainFuncOp(ModuleOp module);
i32_array_t getI32Array(ArrayAttr arrayAttr);
i32_array_t getI32Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int32_t default_value);
i64_array_t getI64Array(ArrayAttr arrayAttr);
i64_array_t getI64Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        int64_t default_value);
f64_array_t getF64Array(ArrayAttr arrayAttr);
f64_array_t getF64Array(std::optional<ArrayAttr> arrayAttr, int64_t num_elem,
                        double default_value);

FuncOp getFuncOp(ModuleOp module, StringRef func_name);
func::CallOp getCallOp(FuncOp func);
llvm::StringRef getName(Operation *op, int index = 0);
llvm::StringRef getName(Value v);
uint32_t getIdx(Value v);
NameLoc getLoc(Value v);
NameLoc getLocLike(Operation *op, llvm::StringRef suffix);
NameLoc getLocLike(Value v, llvm::StringRef suffix);
void setLocSuffix(Operation *op, llvm::StringRef suffix);
void setLoc(Value v, NameLoc loc);
void getInputsOutputs(ModuleOp submodule, std::vector<Value> &inputs,
                      std::vector<Value> &outputs);
void getInputsOutputs(func::CallOp call, std::vector<Value> &inputs,
                      std::vector<Value> &outputs);

bool isNPU_V1();
bool isNPU_V2();

//-----------------------------------------------------------------
// Helper Functions for submodule
//-----------------------------------------------------------------
int getNumSubModule();
std::shared_ptr<std::vector<ModuleOp>> getAllModules();

//-----------------------------------------------------------------
// Helper Functions for weight
//-----------------------------------------------------------------
TensorFile &weightFile();
void setWeightFileName(const std::string &name);
void saveWeight();
void detachWeightFile();

//-----------------------------------------------------------------
// Helper for shape op inference
//-----------------------------------------------------------------
class ShapeHelper {
private:
  ShapeHelper(){};
  ~ShapeHelper(){};
  ShapeHelper(const ShapeHelper &);
  ShapeHelper &operator=(const ShapeHelper &);

public:
  static ShapeHelper &getInstance() {
    static ShapeHelper instance;
    return instance;
  }

  void bindShapeInfo(const Value &v, const std::vector<int64_t> &shape);
  std::vector<int64_t> getShapeInfo(const Value &v);
  bool isShape(const Value &v);

private:
  llvm::DenseMap<Value, std::vector<int64_t>> _shape_info;
};

void bindShapeTensorValue(const Value &v, const std::vector<int64_t> &shape);
std::vector<int64_t> getShapeTensorValue(const Value &v);
bool isShape(const Value &v);


} // namespace module
} // namespace tbc
