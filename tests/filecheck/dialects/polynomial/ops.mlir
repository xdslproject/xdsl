// RUN: XDSL_ROUNDTRIP

// CHECK:      builtin.module {

// CHECK-NEXT:   %[[X_F32:.*]] = "test.op"() : () -> f32
// CHECK-NEXT:   %[[X_F16:.*]] = "test.op"() : () -> f16
// CHECK-NEXT:   %[[X_F64:.*]] = "test.op"() : () -> f64
// CHECK-NEXT:   %[[X_VEC:.*]] = "test.op"() : () -> vector<4xf32>
// CHECK-NEXT:   %[[X_TENSOR:.*]] = "test.op"() : () -> tensor<8xf64>
%x_f32 = "test.op"() : () -> f32
%x_f16 = "test.op"() : () -> f16
%x_f64 = "test.op"() : () -> f64
%x_vec = "test.op"() : () -> vector<4xf32>
%x_tensor = "test.op"() : () -> tensor<8xf64>

// Scalar f32, degree 2, with domain bounds
// CHECK-NEXT:   %{{.*}} = polynomial.eval #polynomial.typed_chebyshev_polynomial<[5.000000e-01 : f64, 1.200000e+00 : f64, 3.000000e-01 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %[[X_F32]] {domain_lower = -1.000000e+00 : f64, domain_upper = 1.000000e+00 : f64, scheme = "clenshaw"} : f32
%r0 = polynomial.eval #polynomial.typed_chebyshev_polynomial<[5.000000e-01 : f64, 1.200000e+00 : f64, 3.000000e-01 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %x_f32 {domain_lower = -1.000000e+00 : f64, domain_upper = 1.000000e+00 : f64, scheme = "clenshaw"} : f32

// Scalar f16, no domain bounds
// CHECK-NEXT:   %{{.*}} = polynomial.eval #polynomial.typed_chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %[[X_F16]] {scheme = "clenshaw"} : f16
%r1 = polynomial.eval #polynomial.typed_chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %x_f16 {scheme = "clenshaw"} : f16

// Scalar f64, custom domain [-10, 0]
// CHECK-NEXT:   %{{.*}} = polynomial.eval #polynomial.typed_chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64, 3.000000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %[[X_F64]] {domain_lower = -1.000000e+01 : f64, domain_upper = 0.000000e+00 : f64, scheme = "clenshaw"} : f64
%r2 = polynomial.eval #polynomial.typed_chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64, 3.000000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %x_f64 {domain_lower = -1.000000e+01 : f64, domain_upper = 0.000000e+00 : f64, scheme = "clenshaw"} : f64

// Vector type, no domain bounds
// CHECK-NEXT:   %{{.*}} = polynomial.eval #polynomial.typed_chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %[[X_VEC]] {scheme = "clenshaw"} : vector<4xf32>
%r3 = polynomial.eval #polynomial.typed_chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %x_vec {scheme = "clenshaw"} : vector<4xf32>

// Tensor type, with domain bounds
// CHECK-NEXT:   %{{.*}} = polynomial.eval #polynomial.typed_chebyshev_polynomial<[5.000000e-01 : f64, 1.500000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %[[X_TENSOR]] {domain_lower = -2.000000e+00 : f64, domain_upper = 2.000000e+00 : f64, scheme = "clenshaw"} : tensor<8xf64>
%r4 = polynomial.eval #polynomial.typed_chebyshev_polynomial<[5.000000e-01 : f64, 1.500000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>, %x_tensor {domain_lower = -2.000000e+00 : f64, domain_upper = 2.000000e+00 : f64, scheme = "clenshaw"} : tensor<8xf64>

// CHECK-NEXT: }
