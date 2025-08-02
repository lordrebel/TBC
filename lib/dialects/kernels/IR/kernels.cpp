

#include "support/module.h"
#include"dialects/kernels/IR/kernels.h"
#include "dialects/kernels/IR/KernelEnum.cpp.inc"

using namespace tbc::kls;
using namespace mlir;
using namespace mlir::func;

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
#include "dialects/kernels/IR/KernelsDialect.cpp.inc"

void KernelDialect::initialize() {

  
  addAttributes<
#define GET_ATTRDEF_LIST
#include "dialects/kernels/IR/KernelAttr.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "dialects/kernels/IR/Kernels.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
//  Kernel Definitions.
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "dialects/kernels/IR/KernelAttr.cpp.inc"
#define GET_OP_CLASSES
#include "dialects/kernels/IR/Kernels.cpp.inc"

