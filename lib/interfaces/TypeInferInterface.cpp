
#include "interfaces/typeInfer_interface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "support/module.h"
#include "support/utils.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>

#include "interfaces/TypeInferInterface.cpp.inc"

namespace tbc {

size_t FloatElementTypeToScore(mlir::Type type) {
  if (isa<mlir::FloatType>(type)) {
    return cast<mlir::FloatType>(type).getWidth();
  }

  llvm_unreachable("Unknown Float Type");
}

size_t IntElementTypeToScore(mlir::Type type) {
  if (isa<mlir::IntegerType>(type)) {
    return cast<mlir::IntegerType>(type).getWidth();
  }
  if (isa<mlir::IndexType>(type)) {
    return 64; // index is 64 bits
  }
  llvm_unreachable("Unknown Integer Type");
}

FloatType ScoreToFloatElementType(size_t score, MLIRContext *ctx) {
  if (score == 128) {
    return mlir::FloatType::getF128(ctx);
  }
  if (score == 64) {
    return mlir::FloatType::getF64(ctx);
  }
  if (score == 32) {
    return mlir::FloatType::getF32(ctx);
  }
  if (score == 16) {
    if (module::getMode() == Mode::BF16 || module::getMode() == Mode::W8BF16 ||
        module::getMode() == Mode::W4BF16) {
      return mlir::FloatType::getBF16(ctx);
    }
    return mlir::FloatType::getF16(ctx);
  }
  if (score == 8) {
    if (module::getMode() == Mode::F8 || module::getMode() == Mode::F8E4M3 ||
        module::getMode() == Mode::W4F8E4M3) {
      return mlir::FloatType::getFloat8E4M3FN(ctx);
    }
    return mlir::FloatType::getFloat8E5M2(ctx);
  }
  llvm_unreachable("Unknown Float Score");
}
// TODO： consider signed and unsigned
mlir::IntegerType ScoreToIntElementType(size_t score, MLIRContext *ctx) {
  if (score == 8) {
    if (module::getMode() == Mode::INT8) {
      return mlir::IntegerType::get(ctx, score, mlir::IntegerType::Signed);
    }
    return mlir::IntegerType::get(ctx, score,
                                  /*isSigned*/ mlir::IntegerType::Unsigned);
  } else {
    return mlir::IntegerType::get(ctx, score);
  }
}

std::vector<size_t> collectInputScore(mlir::Operation *op) {
  std::vector<size_t> scores;
  size_t nums = op->getNumOperands();
  for (auto i = 0; i != nums; i++) {
    auto input = op->getOperand(i);
    if (isa<mlir::NoneType>(input.getType()))
      continue; // Skip NoneType inputs
    Type t = module::getElementType(input);
    if (isa<mlir::IntegerType>(t))
      scores.push_back(IntElementTypeToScore(module::getElementType(input)));
    else if (isa<mlir::FloatType>(t)) {
      scores.push_back(FloatElementTypeToScore(module::getElementType(input)));
    } else {
      llvm_unreachable("Unknown Element Type in collectInputScore");
    }
  }
  return scores;
}

void common_type_inference(mlir::Operation *op) {
  Type type = module::getElementType(op->getOperand(0));
  auto nums = op->getNumResults();
  for (auto i = 0; i < nums; i++) {
    if (isa<mlir::NoneType>(op->getResult(i).getType())) {
      continue; // Skip NoneType results
    }
    module::setElementType(op->getResult(i), type);
  }
}
void broadcast_type_inference(mlir::Operation *op) {
  if (!module::allInputsAreNone(op) &&
      (module::allInputsAreSameElementType(op))) {
    common_type_inference(op);
  }
  std::vector<size_t> scores = collectInputScore(op);
  if (module::allInputsAreFloatElementType(op)) {
    size_t max_score = *std::max_element(scores.begin(), scores.end());
    for (auto i = 0; i < op->getNumResults(); i++) {
      if (isa<mlir::NoneType>(op->getResult(i).getType())) {
        continue; // Skip NoneType results
      }
      module::setElementType(
          op->getResult(i),
          ScoreToFloatElementType(max_score, op->getContext()));
    }
  } else if (module::allInputsAreIntElementType(op)) {
    size_t max_score = *std::max_element(scores.begin(), scores.end());
    for (auto i = 0; i < op->getNumResults(); i++) {
      if (isa<mlir::NoneType>(op->getResult(i).getType())) {
        continue; // Skip NoneType results
      }
      module::setElementType(
          op->getResult(i), ScoreToIntElementType(max_score, op->getContext()));
    }
  } else {
    // TODO:consider sigined and unsigned
    size_t max_score = *std::max_element(scores.begin(), scores.end());
    for (auto i = 0; i < op->getNumResults(); i++) {
      if (isa<mlir::NoneType>(op->getResult(i).getType())) {
        continue; // Skip NoneType results
      }
      module::setElementType(
          op->getResult(i),
          ScoreToFloatElementType(max_score, op->getContext()));
    }
  }
}

} // namespace tbc
