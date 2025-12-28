
#include <iostream>
#include <memory>
#include "ortools/linear_solver/linear_solver.h"

using namespace operations_research;

int main() {
    std::cout << "🚀 OR-Tools Demo - 线性规划求解器\n" << std::endl;
    
    // 创建求解器（使用GLOP - Google的线性优化包）
    std::unique_ptr<MPSolver> solver(MPSolver::CreateSolver("GLOP"));
    if (!solver) {
        std::cerr << "❌ 无法创建求解器！" << std::endl;
        return 1;
    }
    
    std::cout << "✅ 求解器创建成功：" << solver->SolverVersion() << "\n" << std::endl;
    
    // 问题：最大化 3x + 4y
    // 约束条件：
    //   x + 2y <= 14
    //   3x - y >= 0
    //   x - y <= 2
    //   x, y >= 0
    
    // 创建变量
    const double infinity = solver->infinity();
    MPVariable* const x = solver->MakeNumVar(0.0, infinity, "x");
    MPVariable* const y = solver->MakeNumVar(0.0, infinity, "y");
    
    std::cout << "📊 变量数量：" << solver->NumVariables() << std::endl;
    
    // 创建约束条件
    // 约束1：x + 2y <= 14
    MPConstraint* const c0 = solver->MakeRowConstraint(-infinity, 14.0, "c0");
    c0->SetCoefficient(x, 1);
    c0->SetCoefficient(y, 2);
    
    // 约束2：3x - y >= 0
    MPConstraint* const c1 = solver->MakeRowConstraint(0.0, infinity, "c1");
    c1->SetCoefficient(x, 3);
    c1->SetCoefficient(y, -1);
    
    // 约束3：x - y <= 2
    MPConstraint* const c2 = solver->MakeRowConstraint(-infinity, 2.0, "c2");
    c2->SetCoefficient(x, 1);
    c2->SetCoefficient(y, -1);
    
    std::cout << "📏 约束条件数量：" << solver->NumConstraints() << std::endl;
    
    // 创建目标函数：最大化 3x + 4y
    MPObjective* const objective = solver->MutableObjective();
    objective->SetCoefficient(x, 3);
    objective->SetCoefficient(y, 4);
    objective->SetMaximization();
    
    std::cout << "\n🎯 目标函数：最大化 3x + 4y\n" << std::endl;
    
    // 求解
    std::cout << "⏳ 开始求解..." << std::endl;
    const MPSolver::ResultStatus result_status = solver->Solve();
    
    // 输出结果
    if (result_status == MPSolver::OPTIMAL) {
        std::cout << "\n✅ 找到最优解！\n" << std::endl;
        std::cout << "目标函数值 = " << objective->Value() << std::endl;
        std::cout << "x = " << x->solution_value() << std::endl;
        std::cout << "y = " << y->solution_value() << std::endl;
        
        std::cout << "\n📈 求解统计：" << std::endl;
        std::cout << "  迭代次数：" << solver->iterations() << std::endl;
        std::cout << "  求解时间：" << solver->wall_time() << " ms" << std::endl;
    } else {
        std::cout << "\n❌ 未找到最优解！" << std::endl;
        std::cout << "状态码：" << result_status << std::endl;
    }
    
    return 0;
}
