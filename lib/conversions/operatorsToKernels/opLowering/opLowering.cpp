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
        ConcatOpLowering
      >(patterns.getContext());
    }

}
