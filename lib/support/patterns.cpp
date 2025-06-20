//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "support/module.h"
#include "support/patterns.h"

namespace tbc {
namespace patterns {

// if 2 op is same, fuse it.
LogicalResult FuseSameOp::matchAndRewrite(Operation *op,
                                          PatternRewriter &rewriter) const {
  if (isa_and_nonnull<ops::NoneOp, ops::WeightOp>(op)) {
    return failure();
  }
  auto users = op->getUsers();
  auto num_users = std::distance(users.begin(), users.end());
  if (num_users < 2) {
    return failure();
  }
  for (auto first = op->user_begin(); first != op->user_end(); first++) {
    auto it = first;
    for (it++; it != users.end(); it++) {
      if (*first == *it) {
        continue;
      }
      if (module::isSameOp(*first, *it)) {
        if ((*first)->isBeforeInBlock(*it)) {
          (*it)->replaceAllUsesWith(*first);
          rewriter.eraseOp(*it);
        } else {
          (*first)->replaceAllUsesWith(*it);
          rewriter.eraseOp(*first);
        }
        return success();
      }
    }
  }
  return failure();
}

template <typename OpTy>
LogicalResult
FuseRepeatPattern<OpTy>::matchAndRewrite(OpTy op,
                                         PatternRewriter &rewriter) const {
  auto in_op = op.getInput().getDefiningOp();
  if (nullptr == in_op || in_op->hasOneUse() == false) {
    return failure();
  }
  if (!isa<OpTy>(in_op)) {
    return failure();
  }
  op->setOperand(0, in_op->getOperand(0));
  rewriter.eraseOp(in_op);
  return success();

  // handle situations like
  //           -reshape
  // reshape -{
  //           -reshape

  // maybe not for now, just record here

  // std::vector<mlir::Operation *> users;
  // for (auto user: op->getUsers()) {
  //   if (!isa<OpTy>(user)) {
  //     return failure();
  //   }
  //   users.emplace_back(user);
  // }
  // for(auto user: users) {
  //   user->setOperand(0, op->getOperand(0));
  // }
  // rewriter.eraseOp(op);
  // return success();
}

template LogicalResult FuseRepeatPattern<ops::ReshapeOp>::matchAndRewrite(
    ops::ReshapeOp op, PatternRewriter &rewriter) const;

template <typename From, typename To>
LogicalResult
ConvertPattern<From, To>::matchAndRewrite(From op,
                                          PatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<To>(op, op.getOutput().getType(),
                                  op->getOperands(),
                                  std::vector<NamedAttribute>());
  return success();
}

template LogicalResult
ConvertPattern<ops::SqueezeOp, ops::ReshapeOp>::matchAndRewrite(
    ops::SqueezeOp op, PatternRewriter &rewriter) const;

template LogicalResult
ConvertPattern<ops::UnsqueezeOp, ops::ReshapeOp>::matchAndRewrite(
    ops::UnsqueezeOp op, PatternRewriter &rewriter) const;


} // namespace patterns
} // namespace tpu_mlir
