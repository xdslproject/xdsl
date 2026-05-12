// RUN: XDSL_ROUNDTRIP

// CHECK:      builtin.module {

// --- ChebyshevPolynomialAttr (untyped) ---

// CHECK-NEXT:   "test.op"() {poly = #polynomial.chebyshev_polynomial<[5.000000e-01 : f64, 1.200000e+00 : f64, 3.000000e-01 : f64]>} : () -> ()
"test.op"() {poly = #polynomial.chebyshev_polynomial<[5.000000e-01 : f64, 1.200000e+00 : f64, 3.000000e-01 : f64]>} : () -> ()

// CHECK-NEXT:   "test.op"() {poly = #polynomial.chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64]>} : () -> ()
"test.op"() {poly = #polynomial.chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64]>} : () -> ()

// --- RingAttr ---

"test.op"() {
    ring_f32 = #polynomial.ring<coefficientType = f32>,
    ring_f64 = #polynomial.ring<coefficientType = f64>
} : () -> ()

// CHECK-NEXT:    ring_f32 = #polynomial.ring<coefficientType = f32>
// CHECK-SAME:    ring_f64 = #polynomial.ring<coefficientType = f64>

// --- TypedChebyshevPolynomialAttr ---

// CHECK-NEXT:   "test.op"() {poly = #polynomial.typed_chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64, 3.000000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>} : () -> ()
"test.op"() {poly = #polynomial.typed_chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64, 3.000000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>} : () -> ()

// --- HEIR-style aliased input (ring alias + type alias + generic-form attribute) ---

#ring_f64_alias = #polynomial.ring<coefficientType = f64>
!poly_alias = !polynomial.polynomial<ring = #ring_f64_alias>

// CHECK-NEXT: "test.op"() {poly = #polynomial.typed_chebyshev_polynomial<[1.000000e+00 : f64, 2.000000e+00 : f64]> : !polynomial.polynomial<ring = <coefficientType = f64>>} : () -> ()
"test.op"() {poly = #polynomial<typed_chebyshev_polynomial <[1.000000e+00, 2.000000e+00]> : !poly_alias>} : () -> ()

// CHECK-NEXT: }
