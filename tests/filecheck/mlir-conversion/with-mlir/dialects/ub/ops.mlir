// RUN: MLIR_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

// Proves xDSL emits `ub.poison` in a form that upstream `mlir-opt`
// accepts and re-emits identically, in both the pretty and generic forms.

builtin.module {
  // CHECK: %{{.*}} = ub.poison : i32
  %0 = ub.poison : i32

  // CHECK: %{{.*}} = ub.poison : vector<4xi64>
  %1 = ub.poison : vector<4xi64>

  // The long form with an explicit value is accepted on parse, but since
  // `#ub.poison` is the default value, both xDSL and mlir-opt elide it back to
  // the short form (there is no non-default `PoisonAttrInterface` attribute
  // upstream to produce a non-elided spelling).
  // CHECK: %{{.*}} = ub.poison : f32
  %2 = ub.poison <#ub.poison> : f32
}
