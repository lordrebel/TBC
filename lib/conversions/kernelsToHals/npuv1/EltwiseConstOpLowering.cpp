#include "conversions/KernelsToHals/npuV1Lowering.h"
#include "dialects/hals/IR/hals.h"
#include "dialects/hals/IR/halsStructs.h"
#include "dialects/kernels/IR/kernels.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"

namespace tbc::npuv1 {
LogicalResult
EltWiseConstOpLowering::matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                               ConversionPatternRewriter &rewriter) const {
  auto elitwiseOp = cast<tbc::kls::EltWiseConstOp>(op);
  auto loc = elitwiseOp.getLoc();
  auto ori_type = cast_or_null<RankedTensorType>(elitwiseOp.getOutput().getType());
  if (!ori_type) {
    llvm::errs() << "elitwiseOpLowering returnType:" << elitwiseOp.getType() << "\n";
    llvm_unreachable("invalid");
  }
  auto returnType = typeConverter->convertType(ori_type);

  llvm::SmallVector<NamedAttribute, 4> resAttrs;
  for(auto attr:elitwiseOp->getAttrs()){
    if(attr.getName() == "mode"){
      auto mode_val=hals::symbolizeEltwiseMode(cast<StringAttr>(attr.getValue()).getValue()).value();
      resAttrs.push_back(rewriter.getNamedAttr("mode", hals::EltwiseModeAttr::get(op->getContext(),mode_val)));
    }else{
      resAttrs.push_back(attr);
    }
  }
  auto newOp = rewriter.create<tbc::hals::EltwiseConstOp>(loc, returnType, operands,
                                                 resAttrs);
  rewriter.replaceOp(op, newOp.getResult());
  return success();
}
}; // namespace tbc::npuv1
