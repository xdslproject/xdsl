// RUN: XDSL_ROUNDTRIP

// CHECK:      builtin.module {

// --- ChebyshevPolynomialAttr (untyped) ---

// CHECK-NEXT:   "test.op"() {poly = #polynomial.chebyshev_polynomial<[5.000000e-01 : f64, 1.200000e+00 : f64, 3.000000e-01 : f64]>} : () -> ()
"test.op"() {poly = #polynomial.chebyshev_polynomial<[5.000000e-01 : f64, 1.200000e+00 : f64, 3.000000e-01 : f64]>} : () -> ()

// CHECK-NEXT:   "test.op"() {poly = #polynomial.chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64]>} : () -> ()
"test.op"() {poly = #polynomial.chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64]>} : () -> ()

// --- RingAttr ---

// CHECK-NEXT:   "test.op"() {ring = #polynomial.ring<coefficientType = f64>} : () -> ()
"test.op"() {ring = #polynomial.ring<coefficientType = f64>} : () -> ()

// CHECK-NEXT:   "test.op"() {ring = #polynomial.ring<coefficientType = f32>} : () -> ()
"test.op"() {ring = #polynomial.ring<coefficientType = f32>} : () -> ()

// --- PolynomialType ---

// CHECK-NEXT:   "test.op"() {poly_ty = !polynomial.polynomial<ring = <coefficientType = f64>>} : () -> ()
"test.op"() {poly_ty = !polynomial.polynomial<ring = <coefficientType = f64>>} : () -> ()

// --- TypedChebyshevPolynomialAttr ---

// CHECK-NEXT:   "test.op"() {poly = #polynomial.typed_chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64, 3.000000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>} : () -> ()
"test.op"() {poly = #polynomial.typed_chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64, 3.000000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>} : () -> ()

// CHECK-NEXT: }
