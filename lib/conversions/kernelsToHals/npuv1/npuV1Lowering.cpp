#include "conversions/KernelsToHals/npuV1Lowering.h"

namespace tbc::npuv1 {

  void populateKernelsToHalsConversionPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::TypeConverter &typeConverter){
      patterns.add<
        InputOpLowering,
        WeightOpLowering,
        EltWiseOpLowering,
        EltWiseConstOpLowering,
        LutOpLowering,
        ReturnOpLowering,
        Conv2dOpLowering,
        Pool2DOpLowering
      >(typeConverter, patterns.getContext());
    }

} // namespace tbc::npuV1

