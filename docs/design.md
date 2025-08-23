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
|   外部算子    | External Op |      运行在非target上的op，通常是cpu/dsp等                      |                                 无                                          |
|     降级      |                   *lowering*                   |     编译器中将高级抽象转换为低级表示的过程，逐步接近硬件实现     |                                                                       无                                                                       |
|     转换      |                 *conversion*                   |   在不同dialect之间进行IR转换的机制，通过模式匹配将源dialect的操作转换为目标dialect的操作   |                                   [MLIR Conversion文档](https://mlir.llvm.org/docs/DialectConversion/)                                    |
|   转换模式    |              *conversion pattern*              |     定义如何将一个dialect中的操作转换为另一个dialect中操作的规则     |                                                                       无                                                                       |
|   类型转换    |               *type conversion*                |     在dialect转换过程中同时进行的数据类型映射和转换过程     |                                                                       无                                                                       |


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
对`深度学习平台算子的统一抽象`，在这个层级主要是为不同深度学习平台算子服务的，因此并不是特别保证`算子正交`，但是要求尽可能贴合深度学习平台本身算子的语义。在该dialect主要进行shape/type infer，以及和硬件无关的图优化（canonicalize）  

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
对 `硬件平台支持算子的统一抽象`，表示深度学习的算子在硬件平台上的具体实现，在这个层级要求`算子正交`，同时算子类型是固定的因此在 从operator dialect 到 kernel dialect 的lowering 要进行算子的 1对多的转换。 在该dialect上主要进行如下功能：1.一些硬件相关的图优化，2.基于特定硬件的算子融合，3.判定出硬件不支持的算子并将其设定成外部算子（external Op）。     

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

**典型算子示例：**  
（TODO）   


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

**典型算子示例：**  
（TODO）  

## 2.各阶段说明

### converter --> operator dialect

![](./umls/converter2operator.png)  

**功能描述**  
在该阶段主要通过python接口的converter 将 onnx/torch/svjson 模型转换到 operator dialect 上，converter 具体设计请参见*converter-设计*。在完成converter转换后，会给module设置compilephase 为 `OPERATOR_IMPORTED`.  

**相关Pass**  
无  

**相关Conversion**  
无  

**相关Interface**  
无  

### operator dialect --> kernel dialect

![](./umls/operator2kernel.png)  

**功能描述**  

从 operator dialect 到 kernel dialect 之间主要进行硬件无关的图优化，以及和`深度学习平台相关`的图优化.然后就是通过`shapeInferPass` 推理出各op的输入输出的shape，通过 `typeInferPass`推理出各op的输入输出的dtype，完成此阶段后，module的compilePhase会被设置为`OPERATOR_OPTED`。

**相关Pass**  

*PlatformDepedentPass*: 针对不同的学习平台需要进行的图优化，比如tflite 的 op 会有fuseRelu的属性，在此pass中就需要将fused的Relu释放出来，详细设计见：*Pass设计-功能Pass-PlatformDepedentPass*  
*shapeInferPass*: 用于根据模型输入的shape推导出每个op的输出shape，详细设计见：*Pass设计-功能Pass-shapeInferPass*  
*typeInferPass*: 用于根据模型输入的Dtype推导出每个op输出的dtype，详细设计见：*Pass设计-功能Pass-typeInferPass*  
*InferPass*： 基于 operator dialect 进行前向推理的pass，详细设计见：*Pass设计-验证Pass-InferPass*  

**相关Conversion**  

*OperatorToKernel*: 参见：*conversion(lowering) 设计-OperatorToKernel*  

**相关Interface**  
*ShapeInferInterface*: 提供接口：`ShapeInference()` 为op的output设置shape，详细设计见：*Inteface设计-shapeInferInterface*     
*TypeInferInterface*: 提供接口：`TypeInference()` 为op的output设置dtype，详细设计见：*Inteface设计-typeInferInterface*  
*InferenceInterface*: 提供接口：`Init()`、`Deinit()`、`ttd::vector<InferParameter> Inference(const std::vector<InferParameter>&)` 实现 operator op的推理，详细设计参见*Inteface设计-InferenceInterface*   


### kernel dialect --> hal dialect
![](./umls/kernel2hal.png)  

**功能描述**  
该阶段实现从kernel dialect lowering到 hal dialect的过程，首先根据用户输入的target（目标硬件）给module设置一个属性表示最终的编译目标硬件平台，然后根据目标硬件平台的不同，调用不同的图优化pass。然后执行kernel融合pass，将支持融合的算子融合到一起。然后对于不同的目标硬件平台，判定哪些算子应当是external Op，并将其转换成externalOp。最后根据不同的目标硬件平台验证kernelOp在当前硬件平台是否能执行，如果不能，则报错推出。  

**相关Pass**  

*TargetAssiginPass*:  根据命令行输入的编译目标为module设置一个属性表示最终编译的目标硬件平台。详细设计见：*Pass设计-功能Pass-TargetAssiginPass*  
*KernelTargetDependentPass*:  执行一些和硬件编译平台相关的图优化pass。详细设计见：*Pass设计-功能Pass-KernelTargetDependentPass*  
*KernelFusePass*:  执行算子融合，包括但不限于针对conv/mpu的和激活融合，以及其他算子和relu融合等。详细设计见：*Pass设计-功能Pass-KernelFusePass*   
*ExternalOnNpuPass*:  判定Op是否需要转成外部算子（externalOp）运行在cpu或其他计算设备上。详细设计见：*Pass设计-功能Pass-ExternalOnNpuPass*  
*KernelTargetVerifyPass*: 验证kernelOp是否符合目标硬件要求。详细设计见：*Pass设计-功能Pass-KernelTargetVerifyPass*     

**相关Interface**  
(TODO)   

**相关Conversion**  
*kernelToHal*: 将 kernel dialect lowering 到 hal dialect，这里会涉及到type的转换，详细设计见：*conversion(lowering) 设计-KernelToHal*  

### hal dialect --> codegen
![](./umls/hal2codegen.png)  

**功能描述**  
该阶段是从hal dialect 最终转换成目标硬件平台的编译结果，首先先进行规范化，目的是使得halop符合目标硬件平台的硬件约束，然后会执行针对特定硬件目标平台的hal dialect层面的相关图优化，然后是LayergroupPass，这个pass主要完成以下功能：首先是算子分组，对组内算子在nnchw维度上进行tiling，使得SRAM能放的下。然后进行算子调度，使得在考虑SRAM能放的下的情况下尽可能的让算子并行，最后对SRAM根据算子调度结果进行内存划分。根据layergroup的结果为每个group生成硬件sync信息。最后对权重进行打包，生成基于硬件的 ISA（用属性存储ISA相关信息），最终通过codegen生成目标硬件相关的编译结果。

**相关Pass**  

*LegalizePass*：对于不同硬件平台对逐个halOp进行legalize，使其满足指定的硬件平台的硬件约束。 *Pass设计-功能Pass-LegalizePass*    
*HalTargetDependentPass*：基于硬件平台相关的图优化。*Pass设计-功能Pass-HalTargetDependentPass*  
*HalTargetVerifyPass*： 验证halop算子是否满足特定硬件平台的硬件约束。*Pass设计-debugPass-HalTargetVerifyPass*  
*LayerGroupPass*： 实现基于内存的算子分组（group）以及chw切分（tiling）+cascade，然后进行算子调度，根据算子调度的结果在SRAM上进行内存划分。详细设计参见：*Pass设计-功能Pass-LayerGroupPass*   
*InsertSync*：  根据layergroup的结果生成硬件sync信息。详细设计参见：*Pass设计-功能Pass-InsertSync*  
*WeightPackPass*：将权重打包起来最终生成独立的权重文件 详细设计参见： *Pass设计-功能Pass-WeightPackPass*  
*AssemblePass*：生成ISA相关的属性放到op中。详细设计参见：*Pass设计-功能Pass-AssemblePass*  

**相关Interface**  
（TODO）  

**相关Conversion**  
*codgen*: 最终生成目标硬件平台的编译结果。详细设计参见： *conversion(lowering) 设计-Codegen 设计*

## 3.详细设计
详细设计图如下:  
![](./umls/detail_general.png)

### dialect 设计
（TODO）  

#### operator dialect
（TODO）  

##### op  
（TODO）  

##### attribute  
（TODO）  

##### type  
（TODO）  

#### kernel dialect  
（TODO）  

##### op  
（TODO）

##### attribute  
（TODO）  

##### type  
（TODO）  

#### hal dialect  
（TODO）  

##### op  
（TODO）  

##### attribute  
（TODO）   

##### type  
（TODO）  

### pass 设计  
pass 一共分成三类：编译阶段的`功能pass`,`debug pass`,用于验证的 `验证pass`.  

**功能Pass**：用于实现特定功能的pass，如canonical pass，legalize pass。  
**debugPass**：用于检查IR是否合法以及输出一些信息用于调试的pass，如 TargetVerifyHalOpPass，TargetVerifyKernelOpPass.  
**验证Pass**： 基于IR运行一些功能达到检查IR是否合法或者给出评估指标，如：inferencePass、CalCyclePass。  

#### 功能pass  

##### PlatformDependentPass 
**用途**  
用于和深度学习平台相关的图优化pass。  

**dialect**  
operator dialect  

**interface**  
无    

**详细设计**  
（TODO）  

##### CannonicalPass
(TODO)  

##### ShapeInferPass
(TODO)    

##### TypeInferpass
(TODO)  

##### TargetAssiginPass
(TODO)  

##### KernelTargetDependentPass  
(TODO)  

##### KernelFusePass  
(TODO)  

##### ExternalOnNpuPass  
(TODO)  

##### LegalizePass  
(TODO)  

##### HalTargetDependentPass  
(TODO)  

##### HalTargetVerifyPass  
(TODO)   

##### LayerGroupPass  
(TODO)  

##### InsertSync
(TODO)   

##### WeightPackPass
(TODO)     

##### AssemblePass
(TODO)   

#### Debug pass  


##### TargetVerifyHalOpPass
**用途**  

用于验证hal op是否满足当前指定的编译target的硬件约束。  

**dialect**  
Hal dialect  

**interface**  
TargetVerifyInterface  

**详细设计**  
（TODO）  


##### TargetVerifyKernelOpPass  
**用途**  
用于验证Kernel op当前编译的target是否支持。  

**dialect**  
kernel dialect  

**interface**  
TargetVerifyInterface  

**详细设计**  
（TODO）  

#### 验证pass  
（TODO）  

### conversion(lowering) 设计
conversion/lowering主要包括三个：从operator lowering到 Kernel 的conversion、从kernel lowering到Hal的conversion，以及最后的codegen。

#### OperatorToKernel
（TODO）  

#### KernelToHal
（TODO）  

#### Codegen 设计
（TODO）   

### interface 设计  
(TODO)  

### converter 设计  

相关类图如下图所示：  
![](./umls/converter_design.png)


#### 各模块接口设计  
（TODO）  

### target 设计  
（TODO）   


