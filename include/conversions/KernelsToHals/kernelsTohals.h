
#pragma once
#include "conversions/conversion.h"
// namespace mlir

namespace mlir {
#define GEN_PASS_DECL
#define GEN_PASS_DEF_CONVERTKERNELSTOHALS
#include "conversions/Pass.h.inc"
} //
namespace tbc {
  struct ConvertKernelsToHals : public mlir::impl::ConvertKernelsToHalsBase<ConvertKernelsToHals> {

public:
  void runOnOperation() override;

};
}
