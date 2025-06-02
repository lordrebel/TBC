//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/PatternMatch.h"
// Common Patterns
namespace tbc {
namespace patterns {

// if op0 == op1, remove op1
struct FuseSameOp : public mlir::RewritePattern {
  FuseSameOp(mlir::MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}
  mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const override;
};

// if op =  op + op, fuse to one op. such as top::Reshape
template <typename OpTy>
struct FuseRepeatPattern : public mlir::OpRewritePattern<OpTy> {
  using mlir::OpRewritePattern<OpTy>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(OpTy op,
                                mlir::PatternRewriter &rewriter) const override;
};

// convert op a to op b, not care attributes. such as top::Sequence to
// top::Reshape
template <typename From, typename To>
struct ConvertPattern : public mlir::OpRewritePattern<From> {
  using mlir::OpRewritePattern<From>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(From op,
                                mlir::PatternRewriter &rewriter) const override;
};

} // namespace patterns
} // namespace tbc
