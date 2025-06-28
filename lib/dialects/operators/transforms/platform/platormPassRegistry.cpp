#include"dialects/operators/transforms/platform/platormPassRegistry.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include <functional>
#include <unordered_map>
#define DEBUG_TYPE "platform-pass-registry"

namespace tbc::ops{
  std::mutex PlatformPassRegistry::mutex;
  void PlatformPassRegistry::Initialize(){
    //ensure all things is registed
    CALL_REGIST(onnx)
    CALL_REGIST(torch)
    CALL_REGIST(svjson);
  }
  std::unique_ptr<std::unordered_map<Platform, std::function<void(mlir::PassManager &)>>> PlatformPassRegistry::map_ptr=nullptr;
  std::unordered_map<utils::Platform, std::function<void(mlir::PassManager &)>> & PlatformPassRegistry::GetInstance() {
    std::lock_guard<std::mutex> lock(mutex);
    if(map_ptr==nullptr){
      map_ptr=std::make_unique<std::unordered_map<Platform, std::function<void(mlir::PassManager &)>>>();
    }
    return *map_ptr;
  }
  void PlatformPassRegistry::Regist(utils::Platform platform, std::function<void(mlir::PassManager &)> func) {
    auto && map=GetInstance();
    if (map.find(platform) != map.end()) {
      llvm::errs() << "Platform already registered: " + stringifyPlatform(platform);
      llvm_unreachable("Platform already registered");
    }
    LLVM_DEBUG(llvm::dbgs() <<"regist platform pass for:"<< stringifyPlatform(platform) << "\n";);
    map[platform] = func;
  }
  void PlatformPassRegistry::Get(utils::Platform platform, mlir::PassManager & pm) {
    auto && map=GetInstance();
    if (map.find(platform) == map.end()) {
      llvm::errs()<<"Platform not registered: " + stringifyPlatform(platform)<<"\n";
      llvm_unreachable("Platform not registered");
    }
    map[platform](pm);
  }
}
