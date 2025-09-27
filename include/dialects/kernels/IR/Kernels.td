#ifndef TBC_MLIR_KERNELS_TD
#define TBC_MLIR_KERNELS_TD
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

def CompareModeAttr: AnyStrAttrOf<["Equal","Greater","GreaterOrEqual","Less","LessOrEqual", "NotEqual", "Not", "And", "Or"]>;
def EltwiseModeAttr: AnyStrAttrOf<["Add","Sub","Mul","Div"]>;
def ReduceModeAttr: AnyStrAttrOf<["ReduceMin","ReduceMax","ReduceMean","ReduceSum","ReduceProd"]>;
def ReduceOutTypeAttr: AnyStrAttrOf<["IDX","VAL"]>;

def Kernel_PaddingMode: I32EnumAttr<"PaddingMode",
    "requant mode supported by PadOp",
    [
      I32EnumAttrCase<"no_pad", 0>,
      I32EnumAttrCase<"constant", 1>,
      I32EnumAttrCase<"reflect", 2>,
      I32EnumAttrCase<"edge", 3>,
      I32EnumAttrCase<"symmetric", 4>,
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
      I32EnumAttrCase<"zero", 0>,
      I32EnumAttrCase<"nearest", 1>,
      I32EnumAttrCase<"linear", 2>,
      I32EnumAttrCase<"bilinear", 3>,
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
       [])> ;

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

def Kernel_Conv2dOp : Kernel_Op<"Conv2d"> {
  let summary = "Conv2d operator";
  let description = [{
    The operator for 2d convolution function
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    AnyTensorOrNone:$bias,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    DefaultValuedAttr<I64Attr, "1">:$group,
    OptionalAttr<I64ArrayAttr>:$dilations,
    OptionalAttr<BoolAttr>:$do_winograd,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
     tbc::utils::conv_attr_t parseParam();
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
    CompareModeAttr:$mode,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
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
    CompareModeAttr:$mode,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
  }];
}

def Kernel_ConcatOp: Kernel_Op<"Concat",[]> {
  let summary = "concat operator";

  let description = [{
     The operator for compare function
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    SI32Attr:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyRankedTensor:$output);
  let hasVerifier = 0;
}

def Kernel_EltWiseOp: Kernel_Op<"Eltwise",[]>{
  let summary = "Eltwise operator";

  let description = [{
     The operator for compare function
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$right,
    EltwiseModeAttr:$mode,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
  }];
}

def Kernel_EltWiseConstOp: Kernel_Op<"EltwiseConst",[]> {
  let summary = "Eltwise operator";

  let description = [{
     The operator for compare function
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    F64Attr:$const_val,
    EltwiseModeAttr:$mode,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );

  let results = (outs AnyRankedTensor:$output);
  let extraClassDeclaration = [{
  }];
}

def Kernel_InterpOp : Kernel_Op<"Interp"> {
  let summary = "iterp operator";
  let description = [{
      upsample the input tensor to
  }];
  let arguments = (ins
    AnyTensor:$input,
    Kernel_ResizeModeAttr:$mode,
    DefaultValuedAttr<BoolAttr, "false">:$do_activation,
    DefaultValuedAttr<Kernel_ActiveModeAttr, "tbc::kls::ActiveMode::RELU">:$activation_mode,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyTensor:$output);
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
  let extraClassDeclaration = [{
     bool isRelu();
  }];
  
}

def Kernel_PadOp : Kernel_Op<"Pad"> {
  let summary = "pad operator";
  let description = [{
   This operation pads a tensor according to the paddings you specify.
    paddings is an integer tensor with shape [2, n], where n is the rank of tensor.
    For each dimension D of input, paddings[0, D] indicates how many values to add
    before the contents of tensor in that dimension, and paddings[1, D] indicates
    how many values to add after the contents of tensor in that dimension.
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$paddings,
    DefaultValuedAttr<F64Attr, "0.0">:$val,
    DefaultValuedAttr<Kernel_PaddingModeAttr, "tbc::kls::PaddingMode::constant">:$mode,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyTensor:$output);
}

def Kernel_PermuteOp : Kernel_Op<"Permute"> {
  let summary = "permute operator";
  let description = [{
    This operation permutes the dimensions of a tensor according to the specified order.
    The order is an array of integers that specifies the new order of the dimensions.
    For example, if the input tensor has shape [2, 3, 4] and the order is [2, 0, 1],
    the output tensor will have shape [4, 2, 3].
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$order,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let extraClassDeclaration = [{
    tbc::utils::permute_attr_t parseParam();
  }];
  let hasCanonicalizer = 0;
  let hasVerifier = 0;
  let results = (outs AnyTensor:$output);
}

class Kernel_PoolOp<string mnemonic, list<Trait> traits = []> : Kernel_Op<mnemonic,
  !listconcat(traits, [])> {
  let summary = "pool operator";
  let description = [{
    This operation performs pooling on the input tensor.
    The pooling operation can be either average or max pooling.
    The kernel size and stride can be specified to control the pooling operation.
  }];
  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$kernel_shape,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$pads,
    Kernel_PoolModeAttr:$pool_mode,
    DefaultValuedAttr<Kernel_PaddingModeAttr,"tbc::kls::PaddingMode::no_pad">:$pad_mode,
    DefaultValuedAttr<I64Attr, "0">:$pad_value,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let hasCanonicalizer = 0;
  let hasVerifier = 0;
  let extraClassDeclaration = [{
    tbc::utils::pool_attr_t parseParam();
  }];
  let results = (outs AnyTensor:$output);
}

def Kernel_Pool1DOp:Kernel_PoolOp<"Pool1D">;
def Kernel_Pool2DOp:Kernel_PoolOp<"Pool2D">;

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
    DefaultValuedAttr<ReduceOutTypeAttr,"\"VAL\"">:$out_type,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 0;
  let hasVerifier = 1;
}

def Kernel_SliceOp : Kernel_Op<"Slice"> {
  let summary = "Slice operation";
  let description = [{
    Extracts a slice from the input tensor according to the specified start and end indices.
    The slice is defined by the start and end indices for each dimension.
  }];
  let arguments = (ins
    AnyTensor:$input,
    AnyTensorOrNone:$offsetT,
    AnyTensorOrNone:$endsT,
    AnyTensorOrNone:$stepsT,
    I64ArrayAttr:$offset,
    I64ArrayAttr:$steps,
    I64ArrayAttr:$ends,
    DefaultValuedAttr<I64ArrayAttr, "{1}">:$axes,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit
  );
  let results = (outs AnyTensor:$output);
  let hasCanonicalizer = 0;
  let hasVerifier = 0;
  let extraClassDeclaration = [{
    tbc::utils::slice_attr_t parseParam();
  }];
}



#endif
