#pragma once
#include "conversions/conversion.h"
namespace tbc::npuv2 {
  void populateKernelsToHalsConversionPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::TypeConverter &typeConverter);
} // namespace tbc::npuV2
