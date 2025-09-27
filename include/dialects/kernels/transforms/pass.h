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
#include "dialects/kernels/IR/kernels.h"


namespace tbc {
namespace kls {
using mlir::ModuleOp;
std::unique_ptr<mlir::OperationPass<ModuleOp>> createAssginTargetPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "dialects/kernels/transforms/Passes.h.inc"

} // namespace top
} // namespace tpu_mlir
