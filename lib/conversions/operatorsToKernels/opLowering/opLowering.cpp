#include "conversions/OperatorsToKernels/opLowering.h"

namespace tbc::ops{

  void populateOperatorsToKernelsConversionPatterns(
    mlir::RewritePatternSet &patterns){
      patterns.add<
        AddOpLowering,
        AddConstOpLowering,
        MulOpLowering,
        MulConstOpLowering,
        SubOpLowering,
        SubConstOpLowering,
        DivOpLowering,
        ReluOpLowering,
        ActivationLutOpLowering,
        ConcatOpLowering,
        ConvOpLowering,
        PadOpLowering,
        MaxPoolOpLowering
      >(patterns.getContext());
    }

}
