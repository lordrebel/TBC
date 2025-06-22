//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

// =============================================================================
//
// Defines Utils enmus.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_MODULE_ATTR
#define TPU_MLIR_MODULE_ATTR

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinDialect.td"

def Module_Phase: I32EnumAttr<"CompilePhase", "module state",[
      I32EnumAttrCase<"IMPORTED", 0>,
      I32EnumAttrCase<"OPERATOR_OPTED", 1>,
      I32EnumAttrCase<"KERNEL", 2>,
      I32EnumAttrCase<"KERNEL_OPTED", 3>,
      I32EnumAttrCase<"HAL", 4>,
      I32EnumAttrCase<"HAL_OPTED", 5>,
      I32EnumAttrCase<"HAL_ADDRESSED", 6>,
      I32EnumAttrCase<"CODEGEN", 7>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tbc::utils";
}

def Module_Target: I32EnumAttr<"Target", "module target",[
      I32EnumAttrCase<"ALL", 0>,
      I32EnumAttrCase<"NPU_V1", 1000,"npu_v1">,
      I32EnumAttrCase<"NPU_V2", 1002,"npu_v2">,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tbc::utils";
}

def Module_Mode: I32EnumAttr<"Mode", "module mode",[
      I32EnumAttrCase<"INT8",  0>,
      I32EnumAttrCase<"UINT8", 1>,
      I32EnumAttrCase<"INT4",  2>,
      I32EnumAttrCase<"BF16",  3>,
      I32EnumAttrCase<"F16",   4>,
      I32EnumAttrCase<"F32",   5>,
      I32EnumAttrCase<"W8F16", 6>,
      I32EnumAttrCase<"W8BF16",7>,
      I32EnumAttrCase<"W4F16", 8>,
      I32EnumAttrCase<"W4BF16",9>,
      I32EnumAttrCase<"F8",10>,
      I32EnumAttrCase<"F8E4M3",11>,
      I32EnumAttrCase<"F8E5M2",12>,
      I32EnumAttrCase<"W4F8E4M3",13>,
      I32EnumAttrCase<"W4F8E5M2",14>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tbc::utils";
}

def PaddingMode: I32EnumAttr<"PaddingMode",
    "requant mode supported by PadOp",
    [
      I32EnumAttrCase<"no_pad", 0>,
      I32EnumAttrCase<"constant", 1>,
      I32EnumAttrCase<"reflect", 2>,
      I32EnumAttrCase<"symmetric", 3>,
      I32EnumAttrCase<"edge", 4>,
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::tbc::utils";
}

def DataType: I32EnumAttr<"DataType", "represent datatype",[
      I32EnumAttrCase<"INT8",  0>,
      I32EnumAttrCase<"UINT8", 1>,
      I32EnumAttrCase<"INT32",  2>,
      I32EnumAttrCase<"UINT32",  3>,
      I32EnumAttrCase<"INT16",  4>,
      I32EnumAttrCase<"UINT16",  5>,
      I32EnumAttrCase<"INT64",  6>,
      I32EnumAttrCase<"UINT64",  7>,
      I32EnumAttrCase<"BOOL",  8>,
      I32EnumAttrCase<"INT4",  9>,
      I32EnumAttrCase<"UINT4", 10>,
      
      I32EnumAttrCase<"F32",   20>,
      I32EnumAttrCase<"F64",   21>,
      I32EnumAttrCase<"F8E4M3",22>,
      I32EnumAttrCase<"F8E5M2",23>,
      I32EnumAttrCase<"BF16",  24>,
      I32EnumAttrCase<"F16",   25>,
      I32EnumAttrCase<"TF32",    26>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tbc::utils";
}

def Module_Platform: I32EnumAttr<"Platform", "module platform",[
      I32EnumAttrCase<"ONNX",    0>,
      I32EnumAttrCase<"TORCH",   1>,
      I32EnumAttrCase<"TFLITE",  2>,
      I32EnumAttrCase<"SV_JSON",   3>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::tbc::utils";
}

#endif // TPU_MLIR_MODULE_ATTR
