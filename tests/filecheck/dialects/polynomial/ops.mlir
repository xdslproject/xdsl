// RUN: XDSL_ROUNDTRIP

// CHECK:      builtin.module {

// CHECK-NEXT:   %x_f32 = "test.op"() : () -> f32
// CHECK-NEXT:   %x_f16 = "test.op"() : () -> f16
// CHECK-NEXT:   %x_f64 = "test.op"() : () -> f64
// CHECK-NEXT:   %x_vec = "test.op"() : () -> vector<4xf32>
// CHECK-NEXT:   %x_tensor = "test.op"() : () -> tensor<8xf64>
%x_f32 = "test.op"() : () -> f32
%x_f16 = "test.op"() : () -> f16
%x_f64 = "test.op"() : () -> f64
%x_vec = "test.op"() : () -> vector<4xf32>
%x_tensor = "test.op"() : () -> tensor<8xf64>

// Scalar f32, degree 2, default domain [-1, 1]
// CHECK-NEXT:   %r0 = polynomial.eval_chebyshev %x_f32 <[5.000000e-01 : f64, 1.200000e+00 : f64, 3.000000e-01 : f64], -1.000000e+00 : f64, 1.000000e+00 : f64> : f32
%r0 = polynomial.eval_chebyshev %x_f32 <[5.000000e-01 : f64, 1.200000e+00 : f64, 3.000000e-01 : f64], -1.000000e+00 : f64, 1.000000e+00 : f64> : f32

// Scalar f16
// CHECK-NEXT:   %r1 = polynomial.eval_chebyshev %x_f16 <[1.000000e+00 : f64, 2.000000e+00 : f64], -1.000000e+00 : f64, 1.000000e+00 : f64> : f16
%r1 = polynomial.eval_chebyshev %x_f16 <[1.000000e+00 : f64, 2.000000e+00 : f64], -1.000000e+00 : f64, 1.000000e+00 : f64> : f16

// Scalar f64, custom domain [-10, 0]
// CHECK-NEXT:   %r2 = polynomial.eval_chebyshev %x_f64 <[1.000000e+00 : f64, 2.000000e+00 : f64, 3.000000e+00 : f64], -1.000000e+01 : f64, 0.000000e+00 : f64> : f64
%r2 = polynomial.eval_chebyshev %x_f64 <[1.000000e+00 : f64, 2.000000e+00 : f64, 3.000000e+00 : f64], -1.000000e+01 : f64, 0.000000e+00 : f64> : f64

// Vector type
// CHECK-NEXT:   %r3 = polynomial.eval_chebyshev %x_vec <[1.000000e+00 : f64, 2.000000e+00 : f64], -1.000000e+00 : f64, 1.000000e+00 : f64> : vector<4xf32>
%r3 = polynomial.eval_chebyshev %x_vec <[1.000000e+00 : f64, 2.000000e+00 : f64], -1.000000e+00 : f64, 1.000000e+00 : f64> : vector<4xf32>

// Tensor type
// CHECK-NEXT:   %r4 = polynomial.eval_chebyshev %x_tensor <[5.000000e-01 : f64, 1.500000e+00 : f64], -2.000000e+00 : f64, 2.000000e+00 : f64> : tensor<8xf64>
%r4 = polynomial.eval_chebyshev %x_tensor <[5.000000e-01 : f64, 1.500000e+00 : f64], -2.000000e+00 : f64, 2.000000e+00 : f64> : tensor<8xf64>

// CHECK-NEXT: }
