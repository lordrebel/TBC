#include"dialects/operators/transforms/platform/platormPassRegistry.h"
#include "support/utils.h"
namespace tbc::ops{

class OnnxPassCollector: public IPlatformPassCollector{
  public:
    virtual void operator() (mlir::PassManager & pm)final {
      //for test
      //pm.addPass(createPrintOpNamePass());

    };
};

PLATFORM_COLLECTOR_REGISTOR(OnnxPassCollector,utils::Platform::ONNX,onnx)
}
