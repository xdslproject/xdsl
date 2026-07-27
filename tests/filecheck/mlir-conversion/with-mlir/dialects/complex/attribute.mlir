// RUN: MLIR_GENERIC_ROUNDTRIP
// RUN: MLIR_ROUNDTRIP

// CHECK: "test.op"() {attr = #complex.number<:f64 1.000000e+00, 0.000000e+00> : complex<f64>} : () -> ()
"test.op"() {
  attr = #complex.number<:f64 1.0, 0.0> : complex<f64>
} : () -> ()

// CHECK: "test.op"() {attr = #complex.number<:f32 1.000000e+00, 0.000000e+00> : complex<f32>} : () -> ()
"test.op"() {
  attr = #complex.number<:f32 1.0, 0.0> : complex<f32>
} : () -> ()
