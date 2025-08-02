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
      "$_self.cast<StringAttr>().getValue() == \"" # !head(cases) # "\"",
      !foreach(case, !tail(cases),
               "$_self.cast<StringAttr>().getValue() == \"" # case # "\""),
      prev, cur, prev # " || " # cur)>,
  "string attribute whose value is " #
    !foldl(/*init*/!head(cases), /*list*/!tail(cases),
           prev, cur, prev # ", or " # cur)>;

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
    "mlir::DenseI64ArrayAttr":$shape,
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
                     "MemorySpace":$memorySpace,
                     "StorageLayout":$layout,
                     "int64_t":$addr,
                     "TensorKind":$kind), [{
      return $_get($_ctxt, mlir::DenseI64ArrayAttr::get($_ctxt,shape), elementType, memorySpace, layout, addr,kind);
    }]>,
    TypeBuilder<(ins "llvm::ArrayRef<int64_t>":$shape,
                     "mlir::Type":$elementType,
                     "MemorySpace":$memorySpace), [{
      return $_get($_ctxt, mlir::DenseI64ArrayAttr::get($_ctxt,shape), elementType, memorySpace, StorageLayout::nchw,
                   /*addr=*/-1, TensorKind::NOT_SET); 
    }]>,
     TypeBuilder<(ins "mlir::TensorType":$ori_tensor), [{
      return $_get($_ctxt,  mlir::DenseI64ArrayAttr::get($_ctxt,ori_tensor.getShape()), ori_tensor.getElementType(), MemorySpace::NOT_SET, StorageLayout::nchw,
                   /*addr=*/-1, TensorKind::NOT_SET); 
    }]>
   ];
  
  let extraClassDeclaration = [{
    int64_t getNumElements() const {
      int64_t num = 1;
      for (auto dim : getShapeVals()) {
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

    llvm::ArrayRef<int64_t> getShapeVals() const {
      return llvm::ArrayRef<int64_t>(getShape().asArrayRef());
    }
    int64_t getRank() const {
      return getShapeVals().size();
    }
    
  }];
}

// 定义 HalTensor 的类型约束
def AnyHalTensor : Type<CPred<"$_self.isa<::tbc::hals::HalTensorType>()">, 
                        "HAL tensor", "::tbc::hals::HalTensorType">;

// 定义任意排名的 HalTensor（类似 AnyRankedTensor）
def AnyRankedHalTensor : Type<CPred<"$_self.isa<::tbc::hals::HalTensorType>() && "
                                   "$_self.cast<::tbc::hals::HalTensorType>().getRank() >= 0">, 
                             "ranked HAL tensor", "::tbc::hals::HalTensorType">;

def AnyHalTensorOrNone: AnyTypeOf<[AnyRankedHalTensor, NoneType]>;

//===----------------------------------------------------------------------===//
// HAL Operations.
//===----------------------------------------------------------------------===//

class Hal_BaseOp<string mnemonic, list<Trait> traits = []> :
    Op<Hal_Dialect, mnemonic, !listconcat(traits,[])> ;

class Hal_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hal_Dialect, mnemonic, !listconcat(traits,
       [])> ;


#endif
