#ifndef TBC_TYPEINTERFACE
#define TBC_TYPEINTERFACE

include "mlir/IR/OpBase.td"
def TypeInferInterface : OpInterface<"TypeInferInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tbc";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          if output shape is dynamic, inference output shape by input shape;
          or verify output shape
        }],
        /*retType=*/"void",
        /*methodName=*/"type_inference",
        /*args=*/(ins)
      >,
  ];
}
#endif
