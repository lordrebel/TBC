

#include "support/module.h"
#include"dialects/hals/IR/hals.h"

using namespace tbc::hals;
using namespace mlir;
using namespace mlir::func;

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
#include "dialects/hals/IR/HalsDialect.cpp.inc"

void HalDialect::initialize() {

  
  addAttributes<
#define GET_ATTRDEF_LIST
#include "dialects/hals/IR/HalAttr.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "dialects/hals/IR/Hals.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "dialects/hals/IR//HalTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
//  hal Definitions.
//===----------------------------------------------------------------------===//
#include "dialects/hals/IR/HalEnum.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "dialects/hals/IR/HalAttr.cpp.inc"
#define GET_OP_CLASSES
#include "dialects/hals/IR/Hals.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "dialects/hals/IR/HalTypes.cpp.inc"

