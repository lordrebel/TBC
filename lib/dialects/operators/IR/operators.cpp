

#include "support/module.h"
#include"dialects/operators/IR/operator.h"

using namespace tbc::ops;
using namespace mlir;
using namespace mlir::func;

//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
#include "dialects/operators/IR/OperatorsDialect.cpp.inc"

void OperatorDialect::initialize() {

  auto *context = getContext();
  context->getOrLoadDialect<mlir::pdl::PDLDialect>();
  context->getOrLoadDialect<mlir::pdl_interp::PDLInterpDialect>();
  
  addAttributes<
#define GET_ATTRDEF_LIST
#include "dialects/operators/IR/operatorAttr.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "dialects/operators/IR/Operators.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
//  Operator Definitions.
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "dialects/operators/IR/operatorAttr.cpp.inc"
#define GET_OP_CLASSES
#include "dialects/operators/IR/Operators.cpp.inc"

