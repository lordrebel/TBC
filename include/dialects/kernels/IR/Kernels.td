#ifndef TBC_MLIR_KERNELS_TD
#define TBC_MLIR_KERNELS_TD
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "interfaces/ExternalExeInterface.td"

// =============================================================================
//
// Defines Kernel Dialect.
//
//===----------------------------------------------------------------------===//

def Kernel_Dialect : Dialect {
  let name = "kernels";
  let summary = "A tpu dialect for the SOPHGO Deep Learning processors";
  let cppNamespace = "::tbc::kls";
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Kernel Attributes.
//===----------------------------------------------------------------------===//

class Kernel_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Kernel_Dialect, attrName, traits> {
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

def CompareModeAttr: AnyStrAttrOf<["Equal","Greater","GreaterOrEqual","Less","LessOrEqual", "NotEqual", "Not", "And"]>;
def EltwiseModeAttr: AnyStrAttrOf<["Add","Sub","Mul","Div"]>;
def ReduceModeAttr: AnyStrAttrOf<["ReduceMin","ReduceMax","ReduceMean","ReduceSum","ReduceProd"]>;
def ReduceOutTypeAttr: AnyStrAttrOf<["IDX","VAL"]>;

def Kernel_PaddingMode: I32EnumAttr<"PaddingMode",
    "requant mode supported by PadOp",
    [
      I32EnumAttrCase<"constant", 0>,
      I32EnumAttrCase<"reflect", 1>,
      I32EnumAttrCase<"symmetric", 2>,
      I32EnumAttrCase<"edge", 3>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tbc::kls";
}
def Kernel_PaddingModeAttr : EnumAttr<Kernel_Dialect, Kernel_PaddingMode, "Pad_Mode">;

def Kernel_PoolMode: I32EnumAttr<"PoolMode",
    "pooling mode supported by PoolOp",
    [
      I32EnumAttrCase<"Avg", 0>,
      I32EnumAttrCase<"Max", 1>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tbc::kls";
}
def Kernel_PoolModeAttr : EnumAttr<Kernel_Dialect, Kernel_PoolMode, "pool_mode">;

def Kernel_ActiveMode : I32EnumAttr<"ActiveMode",
    "Activation mode for ActiveOp, for sigmoid/exp, e.g.",
    [
      I32EnumAttrCase<"TANH", 0>,
      I32EnumAttrCase<"SIGMOID", 1>,
      I32EnumAttrCase<"RELU", 2>,
      I32EnumAttrCase<"EXP", 3>,
      I32EnumAttrCase<"ELU", 4>,
      I32EnumAttrCase<"SQRT", 5>,
      I32EnumAttrCase<"SQUARE", 6>,
      I32EnumAttrCase<"RSQRT", 7>,
      I32EnumAttrCase<"ABSVAL", 8>,
      I32EnumAttrCase<"LN", 9>,
      I32EnumAttrCase<"ROUND", 10>,
      I32EnumAttrCase<"CEIL", 11>,
      I32EnumAttrCase<"FLOOR", 12>,
      I32EnumAttrCase<"SIN", 13>,
      I32EnumAttrCase<"COS", 14>,
      I32EnumAttrCase<"IS_FINITE", 15>,
      I32EnumAttrCase<"MISH", 16>,
      I32EnumAttrCase<"SWISH", 17>,
      I32EnumAttrCase<"HSWISH", 18>,
      I32EnumAttrCase<"SILU", 19>,
      I32EnumAttrCase<"ARCSIN", 20>,
      I32EnumAttrCase<"ARCCOS", 21>,
      I32EnumAttrCase<"ARCSINH", 22>,
      I32EnumAttrCase<"ARCCOSH", 23>,
      I32EnumAttrCase<"ARCTANH", 24>,
      I32EnumAttrCase<"SINH", 25>,
      I32EnumAttrCase<"COSH", 26>,
      I32EnumAttrCase<"TAN", 27>,
      I32EnumAttrCase<"SIGN", 28>,
      I32EnumAttrCase<"GELU", 29>,
      I32EnumAttrCase<"ERF", 30>,
      I32EnumAttrCase<"HSIGMOID", 31>,
      I32EnumAttrCase<"LOG_SIGMOID", 32>,
      I32EnumAttrCase<"SOFT_PLUS", 33>,
      I32EnumAttrCase<"SOFT_SIGN", 34>,
      I32EnumAttrCase<"LOG2", 35>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tbc::kls";
}
def Kernel_ActiveModeAttr : EnumAttr<Kernel_Dialect, Kernel_ActiveMode, "active_mode">;

def Kernel_ResizeMode : I32EnumAttr<"ResizeMode",
    "Resize mode",
    [
      I32EnumAttrCase<"nearest", 0>,
      I32EnumAttrCase<"linear", 1>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tbc::kls";
}
def Kernel_ResizeModeAttr : EnumAttr<Kernel_Dialect, Kernel_ResizeMode, "mode">;

def Kernel_LutAttr : Kernel_Attr<"LutAttr", "lutattr"> {
  let summary = "Structure of layer group parameters";
  let parameters = (ins
    "int32_t":$sig,
    "int32_t":$bin,
    "int32_t":$cal
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Kernel Types.
//===----------------------------------------------------------------------===//

def AnyTensorOrNone: AnyTypeOf<[AnyRankedTensor, NoneType]>;

//===----------------------------------------------------------------------===//
// Kernel Operations.
//===----------------------------------------------------------------------===//

class Kernel_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Kernel_Dialect, mnemonic, !listconcat(traits,[])> ;

class Kernel_Op<string mnemonic, list<Trait> traits = []> :
    Op<Kernel_Dialect, mnemonic, !listconcat(traits,
       [
        DeclareOpInterfaceMethods<ExecuteOnNPU>,
       ])> ;

def Kernel_ActiveOp: Kernel_Op<"Active",[]>{
  let summary = "Active operator";

  let description = [{
     The operator for activation function
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    Kernel_ActiveModeAttr:$mode
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
  }];
}
def Kernel_CompareOp: Kernel_Op<"Compare",[]>{
  let summary = "Compare operator";

  let description = [{
     The operator for compare function
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyTensorOrNone:$right,
    CompareModeAttr:$mode
  );

  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
  }];
}

def Kernel_CompareConstOp: Kernel_Op<"CompareConst",[]>{
  let summary = "Compare operator";

  let description = [{
     The operator for compare function
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    F64Attr:$right,
    CompareModeAttr:$mode
  );

  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
  }];
}

def Kernel_EltWiseOp: Kernel_Op<"Eltwise",[]>{
  let summary = "Eltwise operator";

  let description = [{
     The operator for compare function
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$right,
    EltwiseModeAttr:$mode
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
  }];
}

def Kernel_EltWiseConstOp: Kernel_Op<"EltwiseConst",[]>{
  let summary = "Eltwise operator";

  let description = [{
     The operator for compare function
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    F64Attr:$right,
    EltwiseModeAttr:$mode
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
  }];
}

def Kernel_LutOp : Kernel_Op<"Lut"> {
  let summary = "lut operator";
  let description = [{
    result = lut_y[i] + lut_k[i] * (x - lut_x[i])
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$lut_x,
    AnyTensor:$lut_y,
    AnyTensor:$lut_k,
    Kernel_LutAttr:$lut_attr
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyTensor:$output);
}
def Kernel_ReduceOp : Kernel_Op<"Reduce"> {
  let summary = "Reduce operation";
  let description = [{
    Computes the mean/max/prod/sum of the input tensor's element along the provided axes.
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$axes,
    BoolAttr:$keepdims,
    ReduceModeAttr:$mode,
    DefaultValuedAttr<ReduceOutTypeAttr,"\"VAL\"">:$out_type
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 0;
  let hasVerifier = 1;
}



#endif
