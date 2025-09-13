#include "conversions/OperatorsToKernels/opLowering.h"
#include "conversions/conversion.h"
#include "conversions/OperatorsToKernels/opertorsTokernels.h"
#include "dialects/operators/IR/operator.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "support/module.h"
#include "support/utils.h"
namespace tbc{
  std::unique_ptr<mlir::Pass> createConvertOperatorsToKernels(){
    return std::make_unique<ConvertOperatorsToKernels>();
  }


  void ConvertOperatorsToKernels::runOnOperation(){
    auto module_=getOperation();
    auto target=mlir::ConversionTarget(getContext());
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<tbc::kls::KernelDialect>();
    target.addIllegalDialect<tbc::ops::OperatorDialect>();

    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<mlir::func::ReturnOp>();
    target.addLegalOp<mlir::func::CallOp>();
    target.addLegalOp<mlir::func::FuncOp>();
    target.addLegalOp<tbc::ops::InputOp>();
    target.addLegalOp<tbc::ops::WeightOp>();

    mlir::MLIRContext *ctx_=&getContext();
    mlir::RewritePatternSet patterns(ctx_);
    ops::populateOperatorsToKernelsConversionPatterns(patterns);
    mlir::applyPatternsAndFoldGreedily(module_, std::move(patterns));
    module::setCompilePhase(utils::CompilePhase::KERNEL);

  }
}
