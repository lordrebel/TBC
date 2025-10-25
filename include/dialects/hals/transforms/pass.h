//===-- Passes.h - ----------------------------------- ----------*- C++ -*-===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "dialects/hals/IR/hals.h"


namespace tbc {
namespace hals {
using mlir::ModuleOp;
std::unique_ptr<mlir::OperationPass<ModuleOp>> createAssginTensorInfosPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "dialects/hals/transforms/Passes.h.inc"

} // namespace top
} // namespace tpu_mlir
