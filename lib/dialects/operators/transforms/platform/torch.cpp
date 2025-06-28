#include"dialects/operators/transforms/platform/platormPassRegistry.h"
#include "support/utils.h"
namespace tbc::ops{



class TorchPassCollector: public IPlatformPassCollector{
  public:
    virtual void operator() (mlir::PassManager & pm)final {

    };
};

PLATFORM_COLLECTOR_REGISTOR(TorchPassCollector,Platform::TORCH,torch)
}
