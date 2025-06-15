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
#include "mlir/Dialect/Quant/QuantOps.h"
#include "dialects/operators/IR/operator.h"
#include "mlir/IR/BuiltinOps.h"


namespace tbc {
namespace ops {
using mlir::ModuleOp;
std::unique_ptr<mlir::OperationPass<ModuleOp>> createInitPass();
std::unique_ptr<mlir::OperationPass<ModuleOp>> createDeinitPass();
std::unique_ptr<mlir::OperationPass<ModuleOp>> createShapeInferPass();
std::unique_ptr<mlir::OperationPass<ModuleOp>> createAssignCompilePhasePass();
#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "dialects/operators/transforms/Passes.h.inc"

} // namespace top
} // namespace tpu_mlir
