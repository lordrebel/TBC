#pragma once

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "traits/traits.h"
#include"support/utils.h"

#include "dialects/kernels/IR/KernelEnum.h.inc"
#include "dialects/kernels/IR/KernelsDialect.h.inc"
#define GET_ATTRDEF_CLASSES
#include "dialects/kernels/IR/KernelAttr.h.inc"
#define GET_OP_CLASSES
#include "dialects/kernels/IR/Kernels.h.inc"
