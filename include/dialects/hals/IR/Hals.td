#ifndef TBC_MLIR_HALS_TD
#define TBC_MLIR_HALS_TD
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "interfaces/MemInfoInferInterface.td"
include "traits/Traits.td"
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
  let useDefaultTypePrinterParser = 1;
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
  I32EnumAttrCase<"Add", 0, "Add">,
  I32EnumAttrCase<"Sub", 1, "Sub">,
  I32EnumAttrCase<"Mul", 2, "Mul">,
  I32EnumAttrCase<"Div", 3, "Div">,
  ]> {
  let cppNamespace = "::tbc::hals";
  let genSpecializedAttr = 0;
}
def Hal_EltwiseModeAttr : EnumAttr<Hal_Dialect, Hal_EltwiseMode, "Eltwise_Mode">;

def Hal_ReduceMode : I32EnumAttr<"ReduceMode", "HAL reduce mode", [
  I32EnumAttrCase<"ReduceMin", 0, "ReduceMin">,
  I32EnumAttrCase<"ReduceMax", 1, "ReduceMax">,
  I32EnumAttrCase<"ReduceMean", 2, "ReduceMean">,
  I32EnumAttrCase<"ReduceSum", 3, "ReduceSum">,
  I32EnumAttrCase<"ReduceProd", 4, "ReduceProd">,
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
  I32EnumAttrCase<"NOT_SET", 0, "not-set">,
  I32EnumAttrCase<"DDR", 1, "ddr">,
  I32EnumAttrCase<"L2", 2, "l2">,
  I32EnumAttrCase<"L1", 3, "l1">,
]> {
  let cppNamespace = "::tbc::hals";
  let genSpecializedAttr = 0;
}
def Hal_MemorySpaceAttr : EnumAttr<Hal_Dialect, Hal_MemorySpace, "Memory_Space">;

def Hal_StorageLayout : I32EnumAttr<"StorageLayout", "HAL storage layout", [
  I32EnumAttrCase<"NOT_SET", 0, "not-set">,
  I32EnumAttrCase<"nchw", 1, "nchw">,
  I32EnumAttrCase<"nhwc", 2, "nhwc">,
  I32EnumAttrCase<"nhwc_sparse", 3, "nhwc_sparse">,
  I32EnumAttrCase<"npu_format", 4, "npu_format">,
  I32EnumAttrCase<"stream", 5, "stream">,

]> {
  let cppNamespace = "::tbc::hals";
  let genSpecializedAttr = 0;
}

def Hal_StorageLayoutAttr : EnumAttr<Hal_Dialect, Hal_StorageLayout, "Storage_Layout">;

def Hal_TensorKind : I32EnumAttr<"TensorKind", "HAL tensor kind", [
   I32EnumAttrCase<"NOT_SET", 0, "not-set">,
  I32EnumAttrCase<"IO", 1, "io-tensor">,
  I32EnumAttrCase<"MID", 2, "internal-tensor">,
  I32EnumAttrCase<"WEIGHT", 3, "weight-tensor">,

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
def Hal_HardwareType: I32EnumAttr<"HardwareType", "HAL hardware type", [
  I32EnumAttrCase<"NOT_SET", 0, "not-set">,
  I32EnumAttrCase<"MPU", 1, "mpu">,
  I32EnumAttrCase<"VPU", 2, "vpu">,
  I32EnumAttrCase<"MTE", 3, "mte">,
  I32EnumAttrCase<"SPU", 4, "spu">,
]> {
  let cppNamespace = "::tbc::hals";
  let genSpecializedAttr = 0;
}

def Hal_HardwareInfoAttr : Hal_Attr<"HardwareInfo", "hardwareinfoattr"> {
  let summary = "Structure of hardware info";
  let parameters = (ins
    "tbc::hals::HardwareType":$type,
    "int32_t": $id
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Hal_CascadeAttr: Hal_Attr<"Cascade", "cascadeattr"> {
  let summary = "Structure of cascade parameters";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$block_params, //n,c,h,w
    ArrayRefParameter<"int64_t">:$buffer_addrs,
    DefaultValuedParameter<"bool", "false">:$is_cascade 
  );

  let builders =[

    TypeBuilder<(ins "cascade_param_t":$params), [{
      return $_get($_ctxt, params.block_size,params.buffer_addrs,params.is_cascade); 
    }]> 
  ];

  let extraClassDeclaration = [{
   std::size_t getNumBuffers() const {
     return getBufferAddrs().size();
   }

   tbc::utils::block_attr_t getBlockAttr() const {
     tbc::utils::block_attr_t block;
     if(!getIsCascade()) {
       block.n = -1;
       block.c = -1;
       block.h = -1;
       block.w = -1;
       return block;
     }
     auto params = getBlockParams();
       block.n = params[0];
       block.c = params[1];
       block.h = params[2];
       block.w = params[3];
     
     return block;
   }

   static mlir::LogicalResult verify(
      llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
      llvm::ArrayRef<int64_t> block_params,
      llvm::ArrayRef<int64_t> buffer_addrs,
      bool is_cascade) {
      if ( is_cascade && block_params.size() != 4) {
        return emitError() << "block_params must have exactly 4 elements (n,c,h,w), got " 
                          << block_params.size();
      }
       //TODO more verify?
      return mlir::success();
    }

    static mlir::LogicalResult verify(
      llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
      tbc::hals::cascade_param_t& params){
      if ( params.is_cascade && params.block_size.size() != 4) {
        return emitError() << "block_params must have exactly 4 elements (n,c,h,w), got " 
                          << params.block_size.size();
      }
       //TODO more verify?
      return mlir::success();
    }


    cascade_param_t parse_params() const;

  }];
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

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    DefaultValuedParameter<"MemorySpace", "MemorySpace::NOT_SET">:$memorySpace,
    DefaultValuedParameter<"StorageLayout", "StorageLayout::NOT_SET">:$layout,
    DefaultValuedParameter<"int64_t", "-1">:$addr,
    DefaultValuedParameter<"TensorKind", "TensorKind::NOT_SET">:$kind,
    OptionalParameter<"CascadeAttr">: $cascade

  );
   
  let assemblyFormat = "`<` `[` $shape `]` `x` $elementType `,` `ms` `:` $memorySpace `,` `ly` `:` $layout `,` `addr` `:` $addr `,` `kind` `:` $kind `,` (`cascade` `:` $cascade^)? `>`";
  
  let builders = [

    TypeBuilder<(ins "llvm::ArrayRef<int64_t>":$shape,
                     "mlir::Type":$elementType,
                     "MemorySpace":$memorySpace), [{
      return $_get($_ctxt, shape, elementType, memorySpace, StorageLayout::nchw,
                   /*addr=*/-1, TensorKind::NOT_SET,nullptr); 
    }]>,
     TypeBuilder<(ins "mlir::TensorType":$ori_tensor), [{
      return $_get($_ctxt,  ori_tensor.getShape(), ori_tensor.getElementType(), MemorySpace::NOT_SET, StorageLayout::nchw,
                   /*addr=*/-1, TensorKind::NOT_SET,nullptr); 
    }]>,
    TypeBuilder<(ins "mlir::TensorType":$ori_tensor,"TensorKind":$kind,"StorageLayout":$layout), [{
      return $_get($_ctxt,  ori_tensor.getShape(), ori_tensor.getElementType(), MemorySpace::NOT_SET,layout,
                   /*addr=*/-1, kind,nullptr); 
    }]>,
    TypeBuilder<(ins "mlir::TensorType":$ori_tensor,"TensorKind":$kind,"MemorySpace":$memspace), [{
      return $_get($_ctxt,  ori_tensor.getShape(), ori_tensor.getElementType(), memspace, StorageLayout::nchw,
                   /*addr=*/-1, kind,nullptr); 
    }]>,
     TypeBuilder<(ins "HalTensorType":$ori_tensor,"int64_t":$addr), [{
      return $_get($_ctxt,  ori_tensor.getShape(), ori_tensor.getElementType(), ori_tensor.getMemorySpace(), ori_tensor.getLayout(),
                   addr, ori_tensor.getKind(),ori_tensor.getCascade()); 
    }]>,
    TypeBuilder<(ins "HalTensorType":$ori_tensor,"CascadeAttr":$cascade), [{
      if(cascade.getIsCascade()){
        return $_get($_ctxt,  ori_tensor.getShape(), ori_tensor.getElementType(), ori_tensor.getMemorySpace(), ori_tensor.getLayout(),
                    ori_tensor.getAddr(), ori_tensor.getKind(),cascade); 
      }else{
        return $_get($_ctxt,  ori_tensor.getShape(), ori_tensor.getElementType(), ori_tensor.getMemorySpace(), ori_tensor.getLayout(),
                    ori_tensor.getAddr(), ori_tensor.getKind(),nullptr);
      }
    }]>,
    TypeBuilder<(ins "hal_tensor_params_t":$params), [{
      if(params.cascade.is_cascade){
        return $_get($_ctxt,  params.shape, params.element_type, params.memory_space, params.layout,
                    params.addr, params.kind,CascadeAttr::get($_ctxt, params.cascade)); 
      }else{
        return $_get($_ctxt,  params.shape, params.element_type, params.memory_space, params.layout,
                    params.addr, params.kind,nullptr);
      }
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

    mlir::RankedTensorType GetRankedTensorType() const {
      return mlir::RankedTensorType::get(getShape(), getElementType());
    }
    
    int64_t getMemorySize() const;

    int64_t getRank() const {
      return getShape().size();
    }

    hal_tensor_params_t parse_params() const;
    
  }];
}

// 定义 HalTensor 的类型约束
def AnyHalTensor : Type<CPred<"mlir::isa<::tbc::hals::HalTensorType>($_self)">, 
                        "HAL tensor", "::tbc::hals::HalTensorType">;

def AnyHalTensorOrNone: AnyTypeOf<[AnyHalTensor, NoneType]>;

//===----------------------------------------------------------------------===//
// HAL Operations.
//===----------------------------------------------------------------------===//

class Hal_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Hal_Dialect, mnemonic, !listconcat(traits,[])> ;

class Hal_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hal_Dialect, mnemonic, !listconcat(traits,
       [])> ;


def Hal_InputOp:Hal_BaseOp<"Input"> {
  let summary = "input operator";
  let description = [{
    input data from ddr memory space
  }];
  let arguments = (ins
    AnyHalTensor:$input
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyHalTensor:$output);
}

def Hal_YieldOp : Hal_BaseOp<"Yield", [Terminator, HasParent<"GroupOp">]> {
  let summary = "Yield values to parent operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyHalTensor>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

}
def Hal_GroupOp: Hal_BaseOp<"Group">{
  let summary = "Group operator";
  let description = [{
    A container operation that encapsulates a sequence of operations.
    It can have multiple inputs and outputs, and it yields results to its parent operation.
  }];

  let arguments = (ins
    Variadic<AnyHalTensor>:$inputs
  );

  let results = (outs
    Variadic<AnyHalTensor>:$outputs
  );

    let regions = (region SizedRegion<1>:$body);

  let hasCanonicalizer = 0;
  let hasVerifier = 0;
}

// def Hal_ReturnOp : Hal_BaseOp< "Return", [
//     Pure,                    
//     HasParent<"mlir::func::FuncOp">,
//     Terminator               
//   ]> {
//   let summary = "Custom return operation for operators dialect";
//   let description = [{
//     The `operators.return` operation represents a return operation within a function.
//     It can optionally return values and supports additional metadata attributes.
//   }];

//   let arguments = (ins
//     Variadic<IOHalTensor>:$operands
//   );

//   let results = (outs);

//   // let assemblyFormat = [{
//   //   ($operands^ `:` type($operands))? attr-dict
//   // }];

//   let hasCanonicalizer = 0;

//   let hasVerifier = 0;

// }

def Hal_WeightOp : Hal_BaseOp<"Weight"> {
  let summary = "load weight operator";

  let description = [{
    Load weight from a file. The file should be a valid .npz format file.
    This Op does not take any input, and the location captures the tensor name.
    The Output is an n-dimensional tensor whose type matches
    the tensor type in the .npz file.
  }];

  let arguments = (ins
    OptionalAttr<F64ArrayAttr>:$scale,
    OptionalAttr<BoolAttr>:$do_compress,
    OptionalAttr<I64ArrayAttr>:$allow_split
  );

  let results = (outs AnyHalTensor:$output);
  let hasCanonicalizer = 0;
  let extraClassDeclaration = [{
  template<typename T>
  std::shared_ptr<std::vector<T>> read();
  std::shared_ptr<std::vector<float>> read_as_float();
  std::shared_ptr<std::vector<int32_t>> read_as_int32();
  std::shared_ptr<std::vector<uint8_t>> read_as_byte();
  template<typename T>
  static mlir::Value create(mlir::Operation * OwnerOp,
                            llvm::StringRef suffix,
                            const std::vector<T>& data,
                            mlir::RankedTensorType& type);
  template<typename T>
  mlir::LogicalResult update(const std::vector<T>& data, size_t count);
  mlir::Value clone_bf16(mlir::Operation * OwnerOp, std::string name = "");
  mlir::Value clone_f16(mlir::Operation * OwnerOp);
  mlir::Value clone_int(mlir::Operation *OwnerOp);
  mlir::Value clone_f8e4m3(mlir::Operation *OwnerOp, bool per_channel_scale = false);
  mlir::Value clone_f8e5m2(mlir::Operation *OwnerOp);
  mlir::Value clone(llvm::StringRef suffix);
  mlir::Value split(int begin, int end, int axis, mlir::Type to_type, std::string suffix);
  }];
}

def Hal_PackWeightOp: Hal_BaseOp<"PackWeight"> {
  let summary = "pack weight operator";

  let description = [{
    Pack weight tensor into a compact representation for efficient storage and access.
  }];

  let arguments = (ins
  );
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<AnyHalTensor>:$outputs);
  let hasCanonicalizer = 0;
  let hasVerifier = 0;  //todo
  let extraClassDeclaration = [{
   //TODO
  }];
}

def Hal_TileOp : Hal_BaseOp<"Tile",
                            [MemInfoInferInput2Output,
                            DeclareOpInterfaceMethods<MemInfoInferInterface,["infer_from_input_to_output"]>]> {
  let summary = "tile operator";
  let description = [{
    
  }];
  let arguments = (ins
    AnyHalTensor:$input,
    I64ArrayAttr:$start_idx,  //nchw
    I64ArrayAttr:$slice    //nchw
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyHalTensor:$output);
}

def Hal_MergeOp : Hal_BaseOp<"Merge",
                            [MemInfoInferOutput2Input,
                             DeclareOpInterfaceMethods<MemInfoInferInterface,["infer_from_output_to_input"]>]> {
  let summary = "tile operator";
  let description = [{
    
  }];
  let arguments = (ins
    Variadic<AnyHalTensor>:$inputs,
    SI32Attr:$axis
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyHalTensor:$output);
}

def Hal_ViewOp : Hal_BaseOp<"View",
                    [SupportInplace]> {
  let summary = "view operator";
  let description = [{
    
  }];
  let arguments = (ins
    AnyHalTensor:$input,
    I64ArrayAttr:$new_shape
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyHalTensor:$output);
}


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
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<Hal_HardwareInfoAttr>:$hw_info
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
    F64Attr:$const_val,
    Hal_EltwiseModeAttr:$mode,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<Hal_HardwareInfoAttr>:$hw_info
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
    OptionalAttr<Hal_HardwareInfoAttr>:$hw_info
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
    AnyHalTensor:$input,
    OptionalAttr<Hal_HardwareInfoAttr>:$hw_info
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
    DefaultValuedAttr<BoolAttr, "false">:$negative_bias,
    DefaultValuedAttr<BoolAttr, "false">:$do_relu,
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<Hal_HardwareInfoAttr>:$hw_info
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
    DefaultValuedAttr<F64Attr, "-1.0">:$relu_limit,
    OptionalAttr<Hal_HardwareInfoAttr>:$hw_info
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
    AnyHalTensor:$input,
    OptionalAttr<Hal_HardwareInfoAttr>:$hw_info
  );
  let hasCanonicalizer = 0;
  let results = (outs AnyHalTensor:$output);
}



#endif
