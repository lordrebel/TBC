#pragma once
#include"support/module.h"
#include"mlir/Pass/PassManager.h"
#include "support/utils.h"
#include <functional>
#include<unordered_map>
#include <mutex>

#define PLATFORM_COLLECTOR_REGISTOR(collector,platform,name) \
void Register__##name() { \
    tbc::ops::PlatformPassRegistry::Regist(platform, collector()); \
  } \

#define DECLARE_PLATFORM(name) \
void Register__##name(); \

#define CALL_REGIST(name) \
Register__##name(); \

namespace tbc::ops {

class PlatformPassRegistry {
  public:
    static std::mutex mutex;  // to prevent race condition when registering passes.
    static void Regist(utils::Platform platform,std::function<void( mlir::PassManager &)>);
    static void Get(utils::Platform platform,mlir::PassManager &);
    static std::unordered_map<utils::Platform, std::function<void( mlir::PassManager &)>> &GetInstance();
    static std::unique_ptr<std::unordered_map<utils::Platform, std::function<void( mlir::PassManager &)>>> map_ptr;
    static void Initialize();
  private:
    PlatformPassRegistry() = default;
    PlatformPassRegistry(const PlatformPassRegistry& ) = delete;
    ~PlatformPassRegistry() = default;

};

class IPlatformPassCollector{
  public:
    virtual void operator() (mlir::PassManager &)=0;
};

//declare plateform register
DECLARE_PLATFORM(onnx)
DECLARE_PLATFORM(torch)
DECLARE_PLATFORM(svjson)

//some common pass declare
std::unique_ptr<mlir::OperationPass<ModuleOp>> createPrintOpNamePass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "dialects/operators/transforms/platform/PlatformPass.h.inc"
}
