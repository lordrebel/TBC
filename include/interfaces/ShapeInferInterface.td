
#ifndef TBC_SHAPEINTERFACE
#define TBC_SHAPEINTERFACE

include "mlir/IR/OpBase.td"

def ShapeInferInterface : OpInterface<"ShapeInferInterface"> {
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
        /*methodName=*/"shape_inference",
        /*args=*/(ins)
      >,
  ];
}
#endif // TPU_MLIR_SHAPEINTERFACE
