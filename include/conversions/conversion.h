#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include <memory>
#include "mlir/Pass/Pass.h"
#include "dialects/hals/IR/hals.h"
#include "dialects/kernels/IR/kernels.h"
#include "dialects/operators/IR/operator.h"

#include "conversions/OperatorsToKernels/opertorsTokernels.h"
#include "conversions/KernelsToHals/kernelsTohals.h"


namespace tbc {
  std::unique_ptr<mlir::Pass> createConvertKernelsToHals();
  std::unique_ptr<mlir::Pass> createConvertOperatorsToKernels();
}

namespace mlir {
#define GEN_PASS_DECL
#include "conversions/Pass.h.inc"

#define GEN_PASS_REGISTRATION
#include "conversions/Pass.h.inc"
}


