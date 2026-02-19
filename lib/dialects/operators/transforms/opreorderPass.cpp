#include "dialects/operators/transforms/pass.h"

namespace tbc::ops{

  class OpreorderPass:public OpreorderBase<OpreorderPass>{
    public:
      void runOnOperation() override{
        //TODO
      }

  };

  std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createOpreorderPass(){
    return std::make_unique<OpreorderPass>();

  }

}
