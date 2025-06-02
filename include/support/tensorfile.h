
//===----------------------------------------------------------------------===//
//
// Common utilities for working with tensor files.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "cnpy.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <ctime>
#include <fstream>
#include <set>
#include <string>
#include <system_error>
#include <type_traits>

#include <iomanip>

namespace tbc {

class TensorFile {
public:
  TensorFile(llvm::StringRef filename, bool readOnly, bool newCreate = false);

  ~TensorFile();

  /// update a tensor for weight compress
  /// if the name is not found, return failure()
  template <typename T>
  mlir::LogicalResult updateTensorData(llvm::StringRef name, const T *data,
                                 size_t count);
  /// add a new tensor to file
  /// if the name is already used, return failure()
  template <typename T>
  mlir::LogicalResult addTensor(llvm::StringRef name, const T *data,
                          mlir::RankedTensorType &type, int64_t length = 0);

  template <typename T>
  mlir::LogicalResult addTensor(llvm::StringRef name, const std::vector<T> *data,
                          mlir::RankedTensorType &type);

  /// add a new tensor to file
  /// if the name is already used, return failure()
  template <typename T>
  mlir::LogicalResult addTensor(llvm::StringRef name, const T *data,
                          std::vector<int64_t> &shape);

  template <typename T>
  mlir::LogicalResult addTensor(llvm::StringRef name, const std::vector<T> *data,
                          std::vector<int64_t> &shape);

  mlir::LogicalResult cloneTensor(llvm::StringRef name, llvm::StringRef suffix);

  /// read a tensor from file
  /// if the name is not found, return failure()
  /// type is provided for checking, return failure() if type does not match
  template <typename T>
  mlir::LogicalResult readTensor(llvm::StringRef name, T *data, size_t count);
  
  template <typename T>
  std::unique_ptr<std::vector<T>>
  readTensor(llvm::StringRef name, mlir::RankedTensorType &type);

  /// delete a tensor from file
  /// if the name is not found, return failure()
  mlir::LogicalResult deleteTensor(const llvm::StringRef name);

  void getAllNames(std::set<mlir::StringRef> &names);

  /// read all tensor from file
  template <typename T>
  mlir::LogicalResult readAllTensors(std::vector<std::string> &names,
                               std::vector<std::vector<T> *> &tensors,
                               std::vector<std::vector<int64_t>> &shapes);

  bool changed();
  bool empty();

  template <typename T>
  void colMajorToRowMajor(T &des, const cnpy::NpyArray &src);
  void save(const std::string &file = "");

private:
  /// load the file
  mlir::LogicalResult load(void);

  std::string filename;
  bool readOnly;
  cnpy::npz_t map;
  std::atomic<int> cnt_del = {0};
  std::atomic<int> cnt_add = {0};
  std::atomic<int> cnt_update = {0};
};

} // tbc

