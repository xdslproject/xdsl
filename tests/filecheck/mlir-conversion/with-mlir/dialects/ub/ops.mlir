// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

// Proves xDSL emits `ub.poison` in a form that upstream `mlir-opt`
// accepts and re-emits identically, in both the pretty and generic forms.

builtin.module {
  // CHECK: %{{.*}} = ub.poison : i32
  %0 = ub.poison : i32

  // CHECK: %{{.*}} = ub.poison : vector<4xi64>
  %1 = ub.poison : vector<4xi64>
}
