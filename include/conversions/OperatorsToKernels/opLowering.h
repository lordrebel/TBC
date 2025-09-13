#pragma once
#include "conversions/conversion.h"
namespace tbc::ops {
void populateOperatorsToKernelsConversionPatterns(
    mlir::RewritePatternSet &patterns);
} // namespace tbc::ops
