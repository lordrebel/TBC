#include "dialects/kernels/transforms/target/targetRegistry.h"
#include "support/log.h"
namespace tbc::kls {
std::mutex TargetDependentPassRegistry::mutex;
std::unique_ptr<TargetDependentPassRegistry>
    TargetDependentPassRegistry::instance = nullptr;
TargetDependentPassRegistry * TargetDependentPassRegistry::GetInstance() {
  std::lock_guard<std::mutex> lock(mutex);
  if (instance == nullptr) {
    instance = std::unique_ptr<TargetDependentPassRegistry>(
        new TargetDependentPassRegistry());
  }
  return instance.get();
}

void TargetDependentPassRegistry::Regist(
    utils::Target target,
    const  std::function<void(mlir::PassManager &)> &pass_collector) {
      auto regist=GetInstance();
      if(regist->map_.find(target)!=regist->map_.end()){
        LOGD<<"TargetDependentPassRegistry::Regist target "<<utils::stringifyTarget(target)<<" has been registered,overriding it.";
      }
      regist->map_[target]=pass_collector;
    }
void TargetDependentPassRegistry::Get(utils::Target target,
                                      mlir::PassManager &pm) {

  auto regist = GetInstance();
  if(regist->map_.count(target)==0){
    LOGE<<"TargetDependentPassRegistry::Get target "<<utils::stringifyTarget(target)<<" not registered";
    llvm_unreachable("TargetDependentPassRegistry::Get target not registered");
  }
  regist->map_[target](pm);
  }

} // namespace tbc::kernels
