
#pragma once

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "interfaces/shapeInfer_interface.h"

#include "traits/traits.h"
#include"support/utils.h"

#include "dialects/operators/IR/OperatorsDialect.h.inc"
#define GET_ATTRDEF_CLASSES
#include "dialects/operators/IR/operatorAttr.h.inc"
#define GET_OP_CLASSES
#include "dialects/operators/IR/Operators.h.inc"
