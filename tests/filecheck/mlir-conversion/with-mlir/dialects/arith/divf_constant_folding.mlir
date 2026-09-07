// Check MLIR's canonicalizer against the same expectations as xDSL. The final
// xdsl-opt invocation only normalizes printing; it does not run any passes.
// RUN: $XDSL_MLIR_OPT --canonicalize "%S/../../../../dialects/arith/divf_constant_folding.mlir" | xdsl-opt | filecheck "%S/../../../../dialects/arith/divf_constant_folding.mlir"
