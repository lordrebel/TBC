add_mlir_pdll_library(TBCOperatorsCommonConstraintsIncGen
                      CommonConstraints.pdll
                      CommonConstraints.h.inc)

add_mlir_pdll_library(TBCOperatorsCommonRewritesIncGen
                      CommonRewrites.pdll
                      CommonRewrites.h.inc)
add_subdirectory(Canonical)
