#pragma once
#include "mlir/IR/OpDefinition.h"

namespace tbc {
namespace traits {
  template <typename ConcreteType>
class MemInfoInferOutput2Input
    : public ::mlir::OpTrait::TraitBase<ConcreteType, MemInfoInferOutput2Input> {};

  template <typename ConcreteType>
class MemInfoInferInput2Output
    : public ::mlir::OpTrait::TraitBase<ConcreteType, MemInfoInferInput2Output> {};

template <typename ConcreteType>
class SupportInplace
    : public ::mlir::OpTrait::TraitBase<ConcreteType, SupportInplace> {};

}  // namespace traits
}
