#ifndef TBC_MEMORYINFO_INFER_INTERFACE
#define TBC_MEMORYINFO_INFER_INTERFACE
include "mlir/IR/OpBase.td"

def MemInfoInferInterface : OpInterface<"MemInfoInferInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tbc";
  let methods = [
    InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"infer_from_input_to_output",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          llvm_unreachable("infer_from_input_to_output not implemented");
        }]
      >,
      InterfaceMethod<
          /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"infer_from_output_to_input",
        /*args=*/(ins),
         /*methodBody=*/"",
        /*defaultImplementation=*/[{
          llvm_unreachable("infer_from_output_to_input not implemented");
        }]
      >,
  ];
  }

#endif
