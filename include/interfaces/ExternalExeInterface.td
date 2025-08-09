
#ifndef TBC_SHAPEINTERFACE
#define TBC_SHAPEINTERFACE

include "mlir/IR/OpBase.td"

def ExecuteOnNPU : OpInterface<"ExecuteOnNPU"> {
  let description = [{
  }];
  let cppNamespace = "::tbc";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          if output shape is dynamic, inference output shape by input shape;
          or verify output shape
        }],
        /*retType=*/"bool",
        /*methodName=*/"IsExecuteOnNPU",
        /*args=*/(ins),
         /*defaultImplementation=*/[{
          // return true
          return true;
        }]

      >,
  ];
}
#endif // TPU_MLIR_SHAPEINTERFACE
