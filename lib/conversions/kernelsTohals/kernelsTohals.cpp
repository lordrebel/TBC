#include "conversions/KernelsToHals/kernelsTohals.h"
#include "conversions/KernelsToHals/npuV1Lowering.h"
#include "conversions/KernelsToHals/npuV2Lowering.h"
#include "conversions/conversion.h"
#include "dialects/kernels/IR/kernels.h"
#include "dialects/operators/IR/operator.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "support/module.h"

namespace tbc {

std::unique_ptr<mlir::Pass> createConvertKernelsToHals() {
  return std::make_unique<ConvertKernelsToHals>();
}

void ConvertKernelsToHals::runOnOperation() {
  auto moduleOp = getOperation();
  mlir::TypeConverter typeConverter;
  typeConverter.addConversion([](mlir::TensorType type) -> mlir::Type {
    return tbc::hals::HalTensorType::get(type.getContext(),
                                         mlir::cast<mlir::TensorType>(type));
  });

  typeConverter.addConversion(
      [](hals::HalTensorType haltensorType) -> mlir::Type {
        return haltensorType;
      });

  typeConverter.addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type resultType,
         mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return nullptr;

        auto inputType = inputs[0].getType();

        if (inputType == resultType) {
          return inputs[0];
        }

        return nullptr;
      });

  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::func::FuncDialect>();
  target.addLegalDialect<tbc::hals::HalDialect>();
  target.addIllegalDialect<tbc::kls::KernelDialect>();
  target.addIllegalDialect<tbc::ops::OperatorDialect>();

  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalOp<mlir::func::CallOp>();
  target.addLegalOp<ops::NoneOp>();
  target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
      [&](mlir::func::ReturnOp op) {
        return typeConverter.isLegal(op.getOperandTypes());
      });
  target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });

  mlir::RewritePatternSet patterns(&getContext());
  // TODO add lowering patterns
  auto compileTarget = module::getTarget();
  if (compileTarget == Target::NPU_V1) {
    tbc::npuv1::populateKernelsToHalsConversionPatterns(patterns,
                                                        typeConverter);
  } else if (compileTarget == Target::NPU_V2) {
    tbc::npuv2::populateKernelsToHalsConversionPatterns(patterns,
                                                        typeConverter);
  } else {
    moduleOp.emitError("unsupport target");
    return signalPassFailure();
  }

  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, typeConverter);
  module::setCompilePhase(utils::CompilePhase::HAL);
  if (mlir::failed(mlir::applyPartialConversion(moduleOp, target,
                                                std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace tbc
