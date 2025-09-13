#pragma once
#include "conversions/conversion.h"
namespace tbc::npuv1 {
  void populateKernelsToHalsConversionPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::TypeConverter &typeConverter);
} // namespace tbc::npuV1
