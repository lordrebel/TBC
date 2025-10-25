#pragma once

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"  

#include "traits/traits.h"
#include"support/utils.h"
#include "support/log.h"

#include "dialects/hals/IR/halsStructs.h"
#include "dialects/hals/IR/HalsDialect.h.inc"
#define GET_ATTRDEF_CLASSES
#include "dialects/hals/IR/HalAttr.h.inc"
#define GET_TYPEDEF_CLASSES
#include "dialects/hals/IR/HalTypes.h.inc"
#define GET_OP_CLASSES
#include "dialects/hals/IR/Hals.h.inc"

