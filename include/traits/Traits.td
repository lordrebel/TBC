
#ifndef TBC_TRAITS_TD
#define TBC_TRAITS_TD

include "mlir/IR/OpBase.td"

class TBCOpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::tbc::traits";
}

// memory info traits
def MemInfoInferOutput2Input : TBCOpTrait<"MemInfoInferOutput2Input">;
def MemInfoInferInput2Output : TBCOpTrait<"MemInfoInferInput2Output">;
def SupportInplace: TBCOpTrait<"SupportInplace">;

#endif // TBC_TRAITS_TD
