#ifndef TBC_LAYERGROUP_INTERFACE
#define TBC_LAYERGROUP_INTERFACE
include "mlir/IR/OpBase.td"

def LayerGroupInferInterface : OpInterface<"LayerGroupInferInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tbc";
  let methods = [
    InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"bool",
        /*methodName=*/"allowSplit",
        /*args=*/(ins "size_t": $axis,
                  "size_t":$operand_id),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return true;
        }]
      >,
      InterfaceMethod<
          /*desc=*/[{
        }],
        /*retType=*/"bool",
        /*methodName=*/"supportGroupTiling",
        /*args=*/(ins),
         /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return true;
        }]
      >,
  ];
  }

#endif
