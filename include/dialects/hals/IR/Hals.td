#ifndef TBC_MLIR_HALS_TD
#define TBC_MLIR_HALS_TD
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// =============================================================================
//
// Defines Hal Dialect.
//
//===----------------------------------------------------------------------===//

def Hal_Dialect : Dialect {
  let name = "hals";
  let summary = "";
  let cppNamespace = "::tbc::hals";
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Hal Attributes.
//===----------------------------------------------------------------------===//

class Hal_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Hal_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

// A string attribute whose value are one of the values in `cases`.
class AnyStrAttrOf<list<string> cases> : StringBasedAttr<
  CPred<!foldl(
      "mlir::cast<StringAttr>($_self).getValue() == \"" # !head(cases) # "\"",
      !foreach(case, !tail(cases),
               "mlir::cast<StringAttr>($_self).getValue() == \"" # case # "\""),
      prev, cur, prev # " || " # cur)>,
  "string attribute whose value is " #
    !foldl(/*init*/!head(cases), /*list*/!tail(cases),
           prev, cur, prev # ", or " # cur)>;

def HaL_CompareMode :I32EnumAttr<"CompareMode", "HAL compare mode", [
  I32EnumAttrCase<"Equal", 0, "equal">,
  I32EnumAttrCase<"Greater", 1, "Greater">,
  I32EnumAttrCase<"Less", 2, "Less">,
  I32EnumAttrCase<"LessOrEqual", 3, "LessOrEqual">,
  I32EnumAttrCase<"NotEqual", 4, "NotEqual">,
  I32EnumAttrCase<"Not", 5, "not">,
  I32EnumAttrCase<"And", 6, "and">,
  I32EnumAttrCase<"Or", 7, "or">,
  ]> {
  let cppNamespace = "::tbc::hals";
  let genSpecializedAttr = 0;
}
def Hal_CompareModeAttr : EnumAttr<Hal_Dialect, HaL_CompareMode, "CompareOpMode" >;

def Hal_EltwiseMode : I32EnumAttr<"EltwiseMode", "HAL eltwise mode", [
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Sub", 1, "sub">,
  I32EnumAttrCase<"Mul", 2, "mul">,
  I32EnumAttrCase<"Div", 3, "div">,
  ]> {
  let cppNamespace = "::tbc::hals";
  let genSpecializedAttr = 0;
}
def Hal_EltwiseModeAttr : EnumAttr<Hal_Dialect, Hal_EltwiseMode, "Eltwise_Mode">;

def Hal_ReduceMode : I32EnumAttr<"ReduceMode", "HAL reduce mode", [
  I32EnumAttrCase<"ReduceMin", 0, "reduce_min">,
  I32EnumAttrCase<"ReduceMax", 1, "reduce_max">,
  I32EnumAttrCase<"ReduceMean", 2, "reduce_mean">,
  I32EnumAttrCase<"ReduceSum", 3, "reduce_sum">,
  I32EnumAttrCase<"ReduceProd", 4, "reduce_prod">,
  ]> {
  let cppNamespace = "::tbc::hals";
  let genSpecializedAttr = 0;
}
def Hal_ReduceModeAttr : EnumAttr<Hal_Dialect, Hal_ReduceMode, "Reduce_Mode">;

def Hal_ReduceOutType : I32EnumAttr<"ReduceOutType", "HAL reduce output type", [
  I32EnumAttrCase<"IDX", 0, "index">,
  I32EnumAttrCase<"VAL", 1, "value">,
]> {
  let cppNamespace = "::tbc::hals";
  let genSpecializedAttr = 0;
}
def Hal_ReduceOutTypeAttr : EnumAttr<Hal_Dialect, Hal_ReduceOutType, "Reduce_Output_Type">;

def Hal_MemorySpace : I32EnumAttr<"MemorySpace", "HAL memory space", [
  I32EnumAttrCase<"NOT_SET", 0, "not set">,
  I32EnumAttrCase<"DDR", 1, "ddr">,
  I32EnumAttrCase<"L2", 2, "l2">,
  I32EnumAttrCase<"L1", 3, "l1">,
]> {
  let cppNamespace = "::tbc::hals";
  let genSpecializedAttr = 0;
}
def Hal_MemorySpaceAttr : EnumAttr<Hal_Dialect, Hal_MemorySpace, "Memory_Space">;

def Hal_StorageLayout : I32EnumAttr<"StorageLayout", "HAL storage layout", [
  I32EnumAttrCase<"nchw", 0, "nchw">,
  I32EnumAttrCase<"nhwc", 1, "nhwc">,
  I32EnumAttrCase<"nhwc_sparse", 2, "nhwc_sparse">,
  I32EnumAttrCase<"npu_format", 3, "npu_format">,
  I32EnumAttrCase<"stream", 4, "stream">,

]> {
  let cppNamespace = "::tbc::hals";
  let genSpecializedAttr = 0;
}

def Hal_StorageLayoutAttr : EnumAttr<Hal_Dialect, Hal_StorageLayout, "Storage_Layout">;

def Hal_TensorKind : I32EnumAttr<"TensorKind", "HAL tensor kind", [
   I32EnumAttrCase<"NOT_SET", 0, "not set">,
  I32EnumAttrCase<"IO", 1, "iotensor">,
  I32EnumAttrCase<"MID", 2, "internal tensor">,
  I32EnumAttrCase<"WEIGHT", 3, "weight tensor">,

]> {
  let cppNamespace = "::tbc::hals";
  let genSpecializedAttr = 0;
}

def Hal_TensorKindAttr : EnumAttr<Hal_Dialect, Hal_TensorKind, "Tensor_Kind">;

def Hal_LutAttr : Hal_Attr<"LutAttr", "lutattr"> {
  let summary = "Structure of layer group parameters";
  let parameters = (ins
    "int32_t":$sig,
    "int32_t":$bin,
    "int32_t":$cal
  );
  let assemblyFormat = "`<` struct(params) `>`";
}
//===----------------------------------------------------------------------===//
// Hals Types.
//===----------------------------------------------------------------------===//
//TODO define HalTensor
def Hal_TensorType : TypeDef<Hal_Dialect, "HalTensor"> {
  let mnemonic = "hal_tensor";
  let summary = "HAL custom tensor type";
  let description = [{
    A custom tensor type for HAL dialect that includes memory space,
    storage layout, and other hardware-specific information.
  }];
  //shape try to use ArrayRefParameter!!!
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    DefaultValuedParameter<"MemorySpace", "MemorySpace::NOT_SET">:$memorySpace,
    DefaultValuedParameter<"StorageLayout", "StorageLayout::nchw">:$layout,
    DefaultValuedParameter<"int64_t", "-1">:$addr,
    DefaultValuedParameter<"TensorKind", "TensorKind::NOT_SET">:$kind

  );
  
  let assemblyFormat = "`<` $shape `x` $elementType `,` $memorySpace `,` $layout `,` $addr `,` $kind `>`";
  
  let builders = [

    TypeBuilder<(ins "llvm::ArrayRef<int64_t>":$shape,
                     "mlir::Type":$elementType,
                     "MemorySpace":$memorySpace), [{
      return $_get($_ctxt, shape, elementType, memorySpace, StorageLayout::nchw,
                   /*addr=*/-1, TensorKind::NOT_SET); 
    }]>,
     TypeBuilder<(ins "mlir::TensorType":$ori_tensor), [{
      return $_get($_ctxt,  ori_tensor.getShape(), ori_tensor.getElementType(), MemorySpace::NOT_SET, StorageLayout::nchw,
                   /*addr=*/-1, TensorKind::NOT_SET); 
    }]>
   ];
  
  let extraClassDeclaration = [{
    int64_t getNumElements() const {
      int64_t num = 1;
      for (auto dim : getShape()) {
        if (dim < 0) return -1; // 动态维度
        num *= dim;
      }
      return num;
    }
    
    bool IsAddrAssigned() const {
      return getAddr() >= 0;
    }
    bool IsTensorKindSetted() const {
      return getKind() !=TensorKind::NOT_SET;
    }
    bool IsOnDDR() const {
      return getMemorySpace() == MemorySpace::DDR;
    }
    bool IsOnL2() const {
      return getMemorySpace() == MemorySpace::L2;
    }
    bool IsOnL1() const {
      return getMemorySpace() == MemorySpace::L1;
    }
    
    int64_t getMemorySize() const;

    int64_t getRank() const {
      return getShape().size();
    }
    
  }];
}

// 定义 HalTensor 的类型约束
def AnyHalTensor : Type<CPred<"mlir::isa<::tbc::hals::HalTensorType>($_self)">, 
                        "HAL tensor", "::tbc::hals::HalTensorType">;

// 定义任意排名的 HalTensor（类似 AnyRankedTensor）
def AnyRankedHalTensor : Type<CPred<"mlir::isa<::tbc::hals::HalTensorType>($_self) && "
                                   "mlir::cast<::tbc::hals::HalTensorType>().getRank() >= 0">, 
                             "ranked HAL tensor", "::tbc::hals::HalTensorType">;

def WeightHalTensor : Type<CPred<"mlir::isa<::tbc::hals::HalTensorType>($_self) && "
                                   "mlir::cast<::tbc::hals::HalTensorType>($_self).IsTensorKindSetted() && "
                                   "mlir::cast<::tbc::hals::HalTensorType>($_self).getKind() == ::tbc::hals::HalTensorKind::WEIGHT">, 
                             "weight HAL tensor", "::tbc::hals::HalTensorType">;

def AnyHalTensorOrNone: AnyTypeOf<[AnyRankedHalTensor, NoneType]>;

//===----------------------------------------------------------------------===//
// HAL Operations.
//===----------------------------------------------------------------------===//

class Hal_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Hal_Dialect, mnemonic, !listconcat(traits,[])> ;

class Hal_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hal_Dialect, mnemonic, !listconcat(traits,
       [])> ;

def Hal_EltwiseOp: Hal_Op<"Eltwise"> {
  let summary = "element-wise operator";
  let description = [{
    result = op1[i] op op2[i]
  }];
  let arguments = (ins
    AnyHalTensor:$input1,
    AnyHalTensor:$input2,
    Hal_EltwiseModeAttr:$opType,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyHalTensor:$output);
}

def Hal_EltwiseConstOp: Hal_Op<"EltwiseConst"> {
  let summary = "element-wise operator";
  let description = [{
    result = op1[i] op op2[i]
  }];
  let arguments = (ins
    AnyHalTensor:$input1,
    F64Attr:$constValue,
    Hal_EltwiseModeAttr:$opType,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyHalTensor:$output);
}

def Hal_LutOp : Hal_Op<"Lut"> {
  let summary = "lut operator";
  let description = [{
    result = lut_y[i] + lut_k[i] * (x - lut_x[i])
  }];
  let arguments = (ins
    AnyHalTensor:$input,
    AnyHalTensor:$lut_x,
    AnyHalTensor:$lut_y,
    AnyHalTensor:$lut_k,
    Hal_LutAttr:$lut_attr,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyHalTensor:$output);
}

def Hal_LoadOp : Hal_Op<"Load"> {
  let summary = "load operator";
  let description = [{
    load data from ddr/l2/l1 memory space
  }];
  let arguments = (ins
    AnyHalTensor:$input
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyHalTensor:$output);
}


def Hal_ScaleOp : Hal_Op<"Scale"> {
  let summary = "scale operator";
  let description = [{
    res=input * scale+ bias
  }];
  let arguments = (ins
    AnyHalTensor:$input,
    F64Attr:$scale,
    F64Attr:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$negative_bias
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyHalTensor:$output);
}

def Hal_ScaleConstOp : Hal_Op<"ScaleConst"> {
  let summary = "scale operator";
  let description = [{
    res=input * scale+ bias
  }];
  let arguments = (ins
    AnyHalTensor:$input,
    AnyHalTensor:$scale,
    AnyHalTensor:$bias,
    DefaultValuedAttr<BoolAttr, "false">:$negative_bias,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyHalTensor:$output);
}

def Hal_StoreOp : Hal_Op<"Store"> {
  let summary = "store operator";
  let description = [{
    store data to ddr/l2/l1 memory space
  }];
  let arguments = (ins
    AnyHalTensor:$input
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyHalTensor:$output);
}

#endif
