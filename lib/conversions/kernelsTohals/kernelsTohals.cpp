#include "conversions/conversion.h"
#include "dialects/kernels/IR/kernels.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "conversions/KernelsToHals/kernelsTohals.h"
#include"support/module.h"
#include "conversions/KernelsToHals/npuV1Lowering.h"
#include "conversions/KernelsToHals/npuV2Lowering.h"

namespace tbc{

  std::unique_ptr<mlir::Pass> createConvertKernelsToHals(){
    return std::make_unique<ConvertKernelsToHals>();
  }

  void ConvertKernelsToHals::runOnOperation(){
    auto moduleOp=getOperation();
    mlir::TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::Type type)->mlir::Type { 
      if(mlir::isa<mlir::TensorType>(type)){
        return tbc::hals::HalTensorType::get(type.getContext(), mlir::cast<mlir::TensorType>(type));
      }
      return type; });
    
      mlir::ConversionTarget target(getContext());
      target.addLegalDialect<mlir::func::FuncDialect>();
      target.addLegalDialect<tbc::hals::HalDialect>();
      target.addLegalDialect<tbc::kls::KernelDialect>();
      
      target.addLegalOp<mlir::ModuleOp>();
      target.addLegalOp<mlir::func::ReturnOp>();
      target.addLegalOp<mlir::func::CallOp>();
      
      target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType());
      });

      mlir::RewritePatternSet patterns(&getContext());
      //TODO add lowering patterns
      auto compileTarget=module::getTarget();
      if(compileTarget==Target::NPU_V1){
        tbc::npuv1::populateKernelsToHalsConversionPatterns(patterns, typeConverter);
      } else if(compileTarget==Target::NPU_V2){
        tbc::npuv2::populateKernelsToHalsConversionPatterns(patterns, typeConverter);
      } else {
        moduleOp.emitError("unsupport target");
        return signalPassFailure();
      }


      mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patterns, typeConverter);
      module::setCompilePhase(utils::CompilePhase::HAL);
      if (mlir::failed(mlir::applyPartialConversion(moduleOp, target, std::move(patterns)))){
        return signalPassFailure();
      }

  }

}
