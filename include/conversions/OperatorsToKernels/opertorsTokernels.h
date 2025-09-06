
#pragma once
#include "conversions/conversion.h"
namespace mlir {
#define GEN_PASS_DECL
#define GEN_PASS_DEF_CONVERTOPERATORSTOKERNELS
#include "conversions/Pass.h.inc"
} //
namespace tbc {

struct ConvertOperatorsToKernels : public mlir::impl::ConvertOperatorsToKernelsBase<ConvertOperatorsToKernels> {

public:
  void runOnOperation() override;

};
  
}
