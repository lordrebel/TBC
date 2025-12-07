#pragma once
#include "conversions/conversion.h"
#include "dialects/kernels/IR/kernels.h"
#include "conversions/KernelsToHals/kernelsTohals.h"
#include "dialects/operators/IR/operator.h"

namespace tbc::npuv1 {
  using namespace mlir;
  using mlir::func::ReturnOp;
  using tbc::ops::InputOp;
  using tbc::ops::WeightOp;
  using namespace tbc::kls;
  //KernelLowering(ReturnOp, 1);
  KernelLowering(InputOp, 1);
  KernelLowering(WeightOp, 1);
  KernelLowering(EltWiseOp, 1);
  KernelLowering(LutOp, 1);
  KernelLowering(EltWiseConstOp, 1);
  KernelLowering(ReturnOp, 1);
  KernelLowering(Conv2dOp, 1);
  KernelLowering(Pool2DOp, 1);



  void populateKernelsToHalsConversionPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::TypeConverter &typeConverter);
} // namespace tbc::npuV1
