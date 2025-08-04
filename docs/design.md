# TBC 设计说明

版本：V0.0.1d

## 术语

|       中文        |                   英文/简写                    |                                            解释                                            |                                                                    参考资料                                                                    |
| :---------------: | :--------------------------------------------: | :----------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
|   深度学习框架    |                   *platform*                   |                 不同的深度学习平台，如 tensorflow/pytorch/tflite/caffe 等                  |                                                                       无                                                                       |
|   多级中间表示    | Multi-Level Intermediate Representation/*mlir* |                           用于构建可重用和可扩展的编译器基础设施                           |                                                       [MLIR官网](https://mlir.llvm.org/)                                                       |
|       方言        |                   *dialect*                    | MLIR中的扩展机制，定义了特定领域的操作、类型和属性，允许在统一框架下表示不同抽象层次的计算 |                                        [MLIR Dialect文档](https://mlir.llvm.org/docs/LangRef/#dialects)                                        |
|       过程        |                     *pass*                     | Pass是编译器中的一个处理单元，它遍历整个程序（或程序的一部分），执行特定的分析或转换操作。 |                                          [MLIR Pass文档](https://mlir.llvm.org/docs/PassManagement/)                                           |
| 模式/模式匹配重写 |   *pattern*/*patternMatch*/*patternRewrite*    |     MLIR中用于定义代码转换规则的机制，通过匹配特定的IR模式并将其重写为等价或优化的形式     |                                   [MLIR Pattern Rewriting文档](https://mlir.llvm.org/docs/PatternRewriter/)                                    |
|     编译阶段      |                 *compilephase*                 |  编译器将源代码转换为目标代码过程中的不同抽象层次，每个阶段代表程序在特定表示形式下的状态  |                                                                       无                                                                       |
|     编译目标      |             compileTarget/*target*             |      编译器最终要生成代码的目标平台或硬件架构，定义了指令集、内存模型、调用约定等特性      |                                                                       无                                                                       |
|      转换器       |                  *converter*                   |                   用于将不同深度学习框架下产生的模型转到统一的mlir表示中                   |                                                                       无                                                                       |
|     代码生成      |           *codegen*/*codegeneration*           |        编译器的最后阶段，将中间表示转换为目标平台的机器码、汇编代码或其他可执行形式        |                                                                       无                                                                       |
|       算子        |                *operator*/*op*                 |       深度学习中的基本计算单元，如卷积、矩阵乘法、激活函数等，定义了计算的语义和接口       |                 [ONNX算子文档](https://onnx.ai/onnx/operators/) [PyTorch算子文档](https://pytorch.org/docs/stable/torch.html)                  |
|       内核        |                    *kernel*                    |               算子在特定硬件上的具体实现，包含了实际的计算逻辑和内存访问模式               |                                                                       无                                                                       |
|    硬件抽象层     |       *HAL*/*Hardware Abstraction Layer*       |                      屏蔽底层硬件差异的抽象层，提供统一的硬件操作接口                      |                                                                       无                                                                       |
|      规范化       |       *canonicalize*/*canonicalization*        |         将IR转换为标准形式的过程，消除冗余操作、简化表达式，使代码更易于分析和优化         |                                   [MLIR Canonicalization文档](https://mlir.llvm.org/docs/Canonicalization/)                                    |
|     算子正交      |            *operator orthogonality*            |  设计原则，要求每个算子具有独立明确的功能，算子间功能不重叠，通过组合基础算子实现复杂功能  | [TVM Relay设计](https://tvm.apache.org/docs/arch/relay_intro.html) [正交性设计原则](https://en.wikipedia.org/wiki/Orthogonality_(programming)) |



## 1.整体结构

整体结构由五层构成：`converter`、`operator dialect`、`kernel dialect`、`hal dialect`,结构示意图如下：  
![](./umls/general.png)  

同时会有八个编译阶段 `IMPORED`, `OPERATOR_OPTED`, `KERNEL`, `KERNEL_OPTED`, `HAL`, `HAL_OPTED`, `HAL_ADDRESSED`, `CODEGEN`。每个阶段具体含义如下：  
1. `IMPORED`: 代表已经从不同的学习框架（platform）转换到当前第一层dialect（i.e. operator dialect）上.  
2. `OPERATOR_OPTED`: 代表在*operator dialect* 上完成了一系列pass变换，已经准备好 lowering 到 *kernel dialect*.  
3. `KERNEL`: 代表已经lowering 到 kernel dialect.  
4. `KERNEL_OPTED`: 代表在*kernel dialect* 上完成了一系列pass变换，已经准备好 lowering 到 *hal dialect*.  
5. `HAL`: 代表已经lowering 到 hal dialect.  
6. `HAL_OPTED`: 代表在hal dialect 上进行了一些target 相关的优化pass.  
7. `HAL_ADDRESSED`: 代表在 hal dialect 上进行了sram的内存划分.  
8. `CODEGEN`: 代表当前在 hal dialect  上已经准备好 codegen.  

### 每层功能简述

#### converter
用于将不同学习平台的模型转换到operator dialect 上，这里用python实现，首先基于mlir的python binding生成我们 operator dialect的 python 接口，然后在其上封装出 mlirImporter，针对不同的学习平台实现不同的converter，而converter的本质行为就是基于mlirImporter 的接口，将不同学习平台的算子转换到我们operator dialect上，这里不一定是一一对应转换，也可以是一对多的转换（一个深度学习的算子对应多个operatorDialect上的算子）

#### operator dialect
对`深度学习平台算子的统一抽象`，在这个层级主要是为不同深度学习平台算子服务的，因此并不是特别保证`算子正交`，但是要求尽可能贴合深度学习平台本身算子的语义。
**设计目标：**  
- **跨平台统一**：为PyTorch、ONNX、TensorFlow、TFLite等不同深度学习框架提供统一的算子表示  
- **语义保真**：最大程度保持原始框架中算子的语义、参数和行为特征  
- **易于转换**：简化从各种深度学习框架到统一IR的转换过程  
- **模型完整性**：确保转换后的模型在功能上与原始模型等价  

**核心特点：**  
- **非正交设计**：允许算子功能重叠，以适应不同框架的设计哲学差异  
  - 例如：PyTorch的`F.hardswish`和手动实现的`x * hardsigmoid(x)`都可以表示  
- **丰富的算子集**：包含各框架的原生算子，避免过早的语义丢失  
- **参数完整性**：保留原始算子的所有参数和属性信息  
- **类型灵活性**：支持动态形状、多种数据类型等框架特性  

**算子分类：**  
1. **基础数学算子**：Add, Mul, MatMul, Conv2D, BatchNorm等  
2. **激活函数**：ReLU, Sigmoid, HardSwish, GELU等  
3. **形状操作**：Reshape, Transpose, Concat, Split等  
4. **池化操作**：MaxPool, AvgPool, AdaptivePool等  
5. **归约操作**：Sum, Mean, Max, Min等  
6. **控制流**：If, Loop, Switch等（如果支持动态图）  

**典型算子示例：**  

```mlir
// 卷积算子 - 保持框架原始语义
%output = operators.Conv2DOp(%input, %weight, %bias) {
  strides = [1, 1],
  padding = [1, 1, 1, 1],
  dilation = [1, 1],
  groups = 1
} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?xf32>) -> tensor<?x?x?x?xf32>

// 复合激活函数 - 直接对应框架算子
%result = operators.HardSwishOp(%input) : tensor<?x?x?x?xf32> -> tensor<?x?x?x?xf32>
```

#### kernel dialect
对 `硬件平台支持算子的统一抽象`，表示深度学习的算子在硬件平台上的具体实现，在这个层级要求`算子正交`，同时算子类型是固定的因此在 从operator dialect 到 kernel dialect 的lowering 要进行算子的 1对多的转换。  
**设计目标：**  
- **计算原语化**：将高级算子分解为硬件友好的基础计算原语  
- **实现策略表达**：明确表示算子的具体计算实现方法  
- **优化空间暴露**：为编译器优化提供更多的分析和变换空间  
- **硬件无关抽象**：描述计算模式而非特定硬件实现  

**核心特点：**  
- **严格正交**：每个算子功能独立，无语义重叠，可自由组合  
- **固定算子集**：预定义的有限基础算子类型，保证编译器分析的完备性  
- **实现导向**：关注"如何计算"而非"计算什么"  
- **组合表达**：复杂操作通过基础算子组合实现  


#### hal dialect
对`硬件层的统一抽象`,表示各具体算子在不同硬件module上的执行模式，因此,在这一层级涉及到了内存划分、权重打包，以及算子合法化（基于硬件约束）等操作。  
**设计目标：**  
- **硬件映射**：将抽象计算映射到具体硬件资源和执行单元  
- **性能优化**：基于目标硬件特性进行深度性能调优  
- **资源管理**：精确控制内存分配、数据搬移和计算调度  
- **约束满足**：确保生成的代码满足硬件的各种约束条件  

**核心特点：**  
- **硬件感知**：深度理解目标硬件的架构特性和限制  
- **执行细节**：描述算子在硬件上的具体执行方式和资源使用  
- **内存层次**：显式管理多级内存层次（寄存器、SRAM、DRAM等）  
- **并发控制**：处理多核、多线程、向量化等并发执行模式  

## 2.各阶段说明

### converter --> operator dialect
（TODO）  

### operator dialect --> kernel dialect
（TODO）  

### kernel dialect --> hal dialect
（TODO）  

### hal dialect --> codegen
（TODO）  

## 3.详细设计
详细设计图如下:  
![](./umls/detail_general.png)

### dialect 设计
（TODO）  

### pass 设计  
pass 一共分成三类：编译阶段的`功能pass`,`debug pass`,用于验证的 `验证pass`.  

#### 功能pass
（TODO）  

#### debug pass
（TODO）  

#### 验证pass
（TODO）  

### converter 设计
（TODO）  

### target 设计
（TODO）  


