#pragma once
#include "support/module.h"
#include "support/utils.h"
#include <mutex>
namespace tbc::kls {
class TargetDependentPassRegistry {
  public:
    static void Regist(utils::Target target, const std::function<void( mlir::PassManager &)> & pass_collector);
    static void Get(utils::Target target, mlir::PassManager & pm);
    ~TargetDependentPassRegistry() = default;
  private:
    static std::mutex mutex;  // to prevent race condition when registering passes.
    static std::unique_ptr<TargetDependentPassRegistry> instance;
    static TargetDependentPassRegistry * GetInstance();
    std::unordered_map<utils::Target, std::function<void( mlir::PassManager &)>> map_;
    TargetDependentPassRegistry() = default;
    TargetDependentPassRegistry(const TargetDependentPassRegistry& ) = delete;


};
}
